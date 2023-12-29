from dataclasses import dataclass, field
from collections import defaultdict
import copy
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.fx import symbolic_trace
from torch.fx._compatibility import compatibility
from typing import Any, Callable, Dict, List, Tuple, NamedTuple, Optional, Set, Union, Iterable
import torch

# def compare_static_args(arg1, arg2):
#     if type(arg1) != type(arg2):
#         return False
#     if isinstance(arg1, Node) and isinstance(arg2, Node):
#         return True
#     if isinstance(arg1, Iterable):
#         if len(arg1) != len(arg2):
#             return False
#         return all([compare_static_args(arg1[j], arg2[j]) for j in range(len(arg1))])
#     return arg1 == arg2


# def static_args_are_equal_old(pn: Node, gn: Node) -> bool:
#     if len(pn.args) != len(gn.args):
#         return False
#     for i in range(len(pn.args)):
#         if not compare_static_args(pn.args[i], gn.args[i]):
#             return False
#     return True

def check_args_are_equal(args1: Union[List, Tuple], args2: Union[List, Tuple]) -> bool:
    if len(args1) != len(args2):
        return False

    for a1, a2 in zip(args1, args2):
        if isinstance(a1, Node) and isinstance(a2, Node):
            matched = True
        elif isinstance(a1, (list, tuple)) and isinstance(a2, (list, tuple)):
            matched = check_args_are_equal(a1, a2)

        if not matched:
            return False

    return True

@compatibility(is_backward_compatible=False)
@dataclass
class InternalMatch():
    # Nodes from which the match was found
    anchors: List[Node]
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: Dict[Node, Node] = field(default_factory=dict)

    # Maps modules in the pattern subgraph to nodes in the larger graph
    modules_map: Dict[Node, Node] = field(default_factory=dict)

    # nodes in target graph that are matched placeholder in pattern
    placeholder_nodes: List[Node] = field(default_factory=list)

    # nodes in matched subgraph returned by output
    returning_nodes: List[Node] = field(default_factory=list)

    # map from a string name to a node in the target graph
    # only available if the matcher is `SubgraphMatcherWithNameNodesMap`
    name_node_map: Dict[str, Node] = field(default_factory=dict)

    def __copy__(self):
        return InternalMatch(anchors=self.anchors, nodes_map=self.nodes_map.copy(),
                             placeholder_nodes=self.placeholder_nodes.copy(),
                             returning_nodes=self.returning_nodes.copy())

@compatibility(is_backward_compatible=False)
class SubgraphMatcher:
    def __init__(self, pattern: Graph,
                 match_output: bool = False,
                 match_placeholder: bool = False,
                 remove_overlapping_matches: bool = True,
                 ignore_literals: bool = False) -> None:
        """
        Args:
            pattern: the targeted matching pattern, represented in fx.Graph.
            match_output: If True, output node in the pattern graph will be treated as a part of the targeted pattern.
                If False, output node is ignored during match.
            match_placeholder: If True, placeholder node in the pattern graph will be treated as a part of
                the targeted pattern. If False, placeholder nodes will be used a wildcard.
            remove_overlapping_matches: If True, in the case of overlapping matches, only the first match
                will be returned.
            ignore_literals: If True, will not check if literals are equal and
                will instead treat them as wildcards.
        """

        self.pattern = pattern
        self.match_output = match_output
        self.match_placeholder = match_placeholder
        self.remove_overlapping_matches = remove_overlapping_matches
        self.ignore_literals = ignore_literals

        if len(pattern.nodes) == 0:
            raise ValueError("SubgraphMatcher cannot be initialized with an empty pattern")

        for node in pattern.nodes:
            if node.op != "output":
                assert len(node.users) > 0, \
                       "SubgraphMatcher cannot be initialized with an pattern with dead code"

        # TODO: assert pattern is a connected graph

        self.pattern_placeholder_nodes = [n for n in pattern.nodes if n.op == "placeholder"]
        output_node = next(iter(reversed(pattern.nodes)))
        # nodes returned by outputs
        self.pattern_returning_nodes: List[Node] = output_node.all_input_nodes

        self.pattern_anchors: List[Node] = []
        if match_output:
            self.pattern_anchors = [output_node]
        else:
            # If a node has output_node as the ONLY user, then this node is a graph sink,
            # and should be matched against as an anchor
            self.pattern_anchors = [n for n in output_node.all_input_nodes if len(n.users) == 1]
    
    def _match_attributes(self, pn: Node, gn: Node) -> bool:
        # Attributes matching is complicated. Right now we only support matching constant tensor
        assert isinstance(pn.target, str), f"pn.target {pn.target} must be a string."
        assert isinstance(gn.target, str), f"gn.target {gn.target} must be a string."
        pn_value = getattr(pn.graph.owning_module, pn.target)
        gn_value = getattr(gn.graph.owning_module, gn.target)
        if type(pn_value) != type(gn_value):
            return False

        # Don't require exact match on tensor values.
        if isinstance(pn_value, torch.Tensor):
            return isinstance(gn_value, torch.Tensor)
        else:
            raise RuntimeError(f"Unsupported type {pn_value} when matching attributes")
        return False
    
    # Add support for module type matching when checking for "equal" nodes
    def _nodes_are_equal(self, pn: Node, gn: Node) -> bool:
        # TODO: match args and kwargs

        # if exact match for placeholder is not required, then use placeholder as a wildcard
        if not self.match_placeholder and pn.op == "placeholder":
            return True

        if pn.op == gn.op:
            if pn.op == "placeholder" or pn.op == "output":
                return True
            elif pn.op == "get_attr":
                return self._match_attributes(pn, gn)
            
            # Handle case of "call_module" and other ops
            # if not static_args_are_equal(pn, gn):
            #     return False
            if not check_args_are_equal(pn.args, gn.args):
                return False

            if pn.op == "call_module":
                pn_submodules = dict(pn.graph.owning_module.named_modules())
                gn_submodules = dict(gn.graph.owning_module.named_modules())

                # Only types should be same
                if type(pn_submodules[pn.target]) == type(gn_submodules[gn.target]):
                    return True

            return pn.target == gn.target
        return False

    def _is_contained(self, nodes_map: Dict[Node, Node]) -> bool:
        # `lookup` represents all the nodes in `original_graph`
        # that are part of `pattern`
        lookup: Dict[Node, Node] = {gn : pn for pn, gn in nodes_map.items()}
        for gn, pn in lookup.items():
            # nodes returned by output are allowed to be used in other areas of the graph
            if pn in self.pattern_returning_nodes:
                continue

            for user in gn.users:
                # If this node has users that were not in `lookup`, then it must leak out of the
                # pattern subgraph
                if user not in lookup:
                    return False
        return True

    def _remove_overlapping_matches(self, matches: List[InternalMatch]) -> List[InternalMatch]:
        non_overlapping_matches: List[InternalMatch] = list()
        nodes_matched: Set[Node] = set()

        for match in matches:
            found_overlap = False
            for pn, gn in match.nodes_map.items():
                if pn.op not in {"placeholder", "output"} and gn in nodes_matched:
                    found_overlap = True
                    break

            if not found_overlap:
                non_overlapping_matches.append(match)
                for pn, gn in match.nodes_map.items():
                    if pn.op not in {"placeholder", "output"}:
                        nodes_matched.add(gn)
        return non_overlapping_matches

    def _match_literals(self, pn: Any, gn: Any, match: InternalMatch) -> bool:
        assert not (isinstance(pn, Node) and isinstance(gn, Node)), "pn and gn cannot both be Node"

        if isinstance(pn, Node) and not isinstance(gn, Node):
            if pn.op == "placeholder":
                # Check if we've already matched these nodes in the current
                # traversal
                if pn in match.nodes_map:
                    return match.nodes_map[pn] == gn

                match.nodes_map[pn] = gn
                return True
            else:
                return False
        elif not isinstance(pn, Node) and isinstance(gn, Node):
            return False
        else:
            return type(gn) == type(pn) and gn == pn
        
    def _match_nodes(self, pn: Node, gn: Node, match: InternalMatch) -> bool:

        # Check if we've already matched these nodes in the current
        # traversal
        if pn in match.nodes_map:
            return match.nodes_map[pn] == gn

        # TODO: use a more efficient way to check if gn is matched before: two-way dict
        if gn in match.nodes_map.values():
            return False

        if not self._nodes_are_equal(pn, gn):
            return False

        # Optimistically mark `pn` as a match for `gn`
        saved_match = copy.copy(match)
        match.nodes_map[pn] = gn

        # Match for "call_module" op
        if pn.op == "call_module":
            match.modules_map[pn.target] = gn.target

        # Placeholder is a wildcard and can be matched with any python object
        # (including list/tuple)
        if pn.op == "placeholder":
            return True

        # Recursively traverse upwards to check if `pn` is a true
        # match for `gn`
        match_found = True

        def _match_args(args1: Union[List, Tuple], args2: Union[List, Tuple]) -> bool:
            if len(args1) != len(args2):
                return False

            for a1, a2 in zip(args1, args2):
                if isinstance(a1, Node) and isinstance(a2, Node):
                    matched = self._match_nodes(a1, a2, match)
                elif isinstance(a1, (list, tuple)) and isinstance(a2, (list, tuple)):
                    matched = _match_args(a1, a2)
                else:
                    matched = self._match_literals(a1, a2, match) or self.ignore_literals

                if not matched:
                    return False

            return True

        # Flatten all args/kwargs into 1 list of args
        pn_args, gn_args = None, None
        if (
            (len(pn.args) != len(gn.args) or list(pn.kwargs.keys()) != list(gn.kwargs.keys())) and
            pn.op == "call_function" and
            isinstance(pn.target, torch._ops.OpOverload)
        ):
            args_schema = pn.target._schema.arguments

            def get_all_arguments(orig_args, orig_kwargs):
                all_args = []
                for i, schema in enumerate(args_schema):
                    if schema.name in orig_kwargs:
                        all_args.append(orig_kwargs[schema.name])
                    elif not schema.kwarg_only and i < len(orig_args):
                        all_args.append(orig_args[i])
                    else:
                        all_args.append(schema.default_value)
                return all_args

            pn_args = get_all_arguments(pn.args, pn.kwargs)
            gn_args = get_all_arguments(gn.args, gn.kwargs)

        elif len(pn.args) == len(gn.args) and list(pn.kwargs.keys()) == list(gn.kwargs.keys()):
            pn_args = list(pn.args)
            gn_args = list(gn.args)
            pn_args.extend(list(pn.kwargs.values()))
            gn_args.extend(list(gn.kwargs.values()))
        else:
            match_found = False

        match_found = (
            match_found and
            pn_args is not None and
            gn_args is not None and
            _match_args(pn_args, gn_args)
        )

        if not match_found:
            # revert to saved_match before matching with current node
            match = copy.copy(saved_match)
            return False

        return True

    def match(self, graph: Graph) -> List[InternalMatch]:
        """
        Returns:
            The matched subgraphs.
            Thre returned subgraph would be fully self-contained, meaning the nodes (except placeholder
            and nodes returned by output) can only be consumed by nodes within the matched subgraph.

        Subgraph pattern matcher is implemented with the backtracking style in the following steps:

        1. We first identify all the anchor nodes in the pattern graph. The anchor nodes
        are the "sinks" (nodes with no user other than the output node) of the pattern graph.
        One pattern graph could have multiple anchors if it has multiple return values.

        2. In the target graph, we identify the potential candidate nodes that can be matched
        with each anchor. These anchor-candidate pairs are the starting points for
        pairwise per-node matching.

        3. For each anchor-candidate pair, we simultaneously traverse backwards (DFS) in both
        pattern and target graphs. For every pattern nodes along traversal path, we compare it
        against the target nodes. In case any comparison failed, the match for this anchor-candidate
        pair fails. A match is found when DFS completes traversing the graph. See `self._match_nodes`
        for more details.

        4. In the case of multiple anchors, every anchor will need to find a match using step 3.
        In addition, the matches found between anchors need to have a common intersection node
        in order for the match to be valid. This is implemented with backtracking. See `backtracking`
        for more details.

        Notice: graph traversal must be done in the reverser order because a tensor can have multiple
        consumers, but can only have a single producer. Only with reverser order, we can we jointly
        traverse the pattern and target graph in a deterministic path.

        Warning: In theory, this backtracking algorithm have an **exponential** time complexity. However,
        in practice, it's unlikely to blow up.

        """

        # find candidate nodes to match with pattern anchors
        match_candidates: Dict[Node, List[Node]] = defaultdict(list)
        for pattern_anchor in self.pattern_anchors:
            for node in graph.nodes:
                if self._nodes_are_equal(pattern_anchor, node):
                    match_candidates[pattern_anchor].append(node)
        match_candidates_list = list(match_candidates.items())
        matches: List[InternalMatch] = []

        def backtracking(anchor_index, match):
            if anchor_index == len(match_candidates_list):
                match.placeholder_nodes = [match.nodes_map[pn] for pn in self.pattern_placeholder_nodes]
                match.returning_nodes = [match.nodes_map[pn] for pn in self.pattern_returning_nodes]
                matches.append(match)
                return

            pattern_anchor, candidate_nodes = match_candidates_list[anchor_index]
            saved_match = copy.copy(match)

            for node in candidate_nodes:
                match_found = self._match_nodes(pattern_anchor, node, match)
                if match_found:
                    # match next anchor
                    backtracking(anchor_index + 1, match)

                # revert to saved_match before matching with current anchor
                match = copy.copy(saved_match)

        match = InternalMatch(anchors=self.pattern_anchors)
        if match_candidates_list:
            backtracking(0, match)

        # filter out the matches where the subgraph is not fully_contained
        matches = [match for match in matches if self._is_contained(match.nodes_map)]

        if self.remove_overlapping_matches:
            matches = self._remove_overlapping_matches(matches)

        return matches


@compatibility(is_backward_compatible=True)
class Match(NamedTuple):
    # Node from which the match was found
    anchor: Node
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: Dict[Node, Node]

@compatibility(is_backward_compatible=False)
@dataclass
class ReplacedPatterns:
    # Node from which the match was found
    anchor: Node
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: Dict[Node, Node]
    # List of nodes that were added into the graph
    replacements: List[Node]

def _replace_attributes(gm: GraphModule, replacement: torch.nn.Module) -> None:
    gm.delete_all_unused_submodules()

    if isinstance(replacement, GraphModule):
        replacement.graph.lint()

    def try_get_attr(gm: torch.nn.Module, target: str) -> Optional[Any]:
        module_path, _, attr_name = target.rpartition(".")
        mod: torch.nn.Module = gm.get_submodule(module_path)
        attr = getattr(mod, attr_name, None)
        return attr

    # def try_get_submod(gm: torch.nn.Module, target: str) -> Optional[Any]:
    #     try:
    #         mod: torch.nn.Module = gm.get_submodule(target)
    #         return mod
    #     except AttributeError:
    #         return None

    for node in gm.graph.nodes:
        if node.op == "get_attr" or node.op == "call_module":
            gm_attr = try_get_attr(gm, node.target)
            replacement_attr = try_get_attr(replacement, node.target)

            # CASE 1: This target already exists as an attribute in our
            # result GraphModule. Whether or not it exists in
            # `replacement`, the existing submodule takes precedence.
            if gm_attr is not None:
                continue

            # CASE 2: The target exists as an attribute in `replacement`
            # only, so we need to copy it over.
            elif replacement_attr is not None:
                new_attr = copy.deepcopy(replacement_attr)
                if isinstance(replacement_attr, torch.nn.Module):
                    gm.add_submodule(node.target, new_attr)
                else:
                    setattr(gm, node.target, new_attr)

            # CASE 3: The target doesn't exist as an attribute in `gm`
            # or `replacement`
            else:
                raise RuntimeError("Attempted to create a \"", node.op,
                                   "\" node during subgraph rewriting "
                                   f"with target {node.target}, but "
                                   "the referenced attribute does not "
                                   "exist in the replacement GraphModule")
        
        # elif node.op == "call_module":
        #     gm_submod = try_get_submod(gm, node.target)
        #     replacement_submod = try_get_submod(replacement, node.target)

        #     # CASE 1: This target already exists as an attribute in our
        #     # result GraphModule. Whether or not it exists in
        #     # `replacement`, the existing submodule takes precedence.
        #     if gm_submod is not None:
        #         continue

        #     # CASE 2: The target exists as an attribute in `replacement`
        #     # only, so we need to copy it over.
        #     elif replacement_submod is not None:
        #         new_submod = copy.deepcopy(replacement_submod)
        #         if isinstance(replacement_submod, torch.nn.Module):
        #             gm.add_submodule(node.target, new_attr)
        #         else:
        #             setattr(gm, node.target, new_attr)

        #     # CASE 3: The target doesn't exist as an attribute in `gm`
        #     # or `replacement`
        #     else:
        #         raise RuntimeError("Attempted to create a \"", node.op,
        #                            "\" node during subgraph rewriting "
        #                            f"with target {node.target}, but "
        #                            "the referenced attribute does not "
        #                            "exist in the replacement GraphModule")


    gm.graph.lint()


@compatibility(is_backward_compatible=True)
def replace_pattern(
    gm: GraphModule,
    pattern: Union[Callable, GraphModule],
    replacement: Union[Callable, GraphModule]
) -> List[Match]:
    """
    Matches all possible non-overlapping sets of operators and their
    data dependencies (``pattern``) in the Graph of a GraphModule
    (``gm``), then replaces each of these matched subgraphs with another
    subgraph (``replacement``).

    Args:
        ``gm``: The GraphModule that wraps the Graph to operate on
        ``pattern``: The subgraph to match in ``gm`` for replacement
        ``replacement``: The subgraph to replace ``pattern`` with

    Returns:
        List[Match]: A list of ``Match`` objects representing the places
        in the original graph that ``pattern`` was matched to. The list
        is empty if there are no matches. ``Match`` is defined as:

        .. code-block:: python

            class Match(NamedTuple):
                # Node from which the match was found
                anchor: Node
                # Maps nodes in the pattern subgraph to nodes in the larger graph
                nodes_map: Dict[Node, Node]

    Examples:

    .. code-block:: python

        import torch
        from torch.fx import symbolic_trace, subgraph_rewriter

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w1, w2):
                m1 = torch.cat([w1, w2]).sum()
                m2 = torch.cat([w1, w2]).sum()
                return x + torch.max(m1) + torch.max(m2)

        def pattern(w1, w2):
            return torch.cat([w1, w2]).sum()

        def replacement(w1, w2):
            return torch.stack([w1, w2])

        traced_module = symbolic_trace(M())

        subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)

    The above code will first match ``pattern`` in the ``forward``
    method of ``traced_module``. Pattern-matching is done based on
    use-def relationships, not node names. For example, if you had
    ``p = torch.cat([a, b])`` in ``pattern``, you could match
    ``m = torch.cat([a, b])`` in the original ``forward`` function,
    despite the variable names being different (``p`` vs ``m``).

    The ``return`` statement in ``pattern`` is matched based on its
    value only; it may or may not match to the ``return`` statement in
    the larger graph. In other words, the pattern doesn't have to extend
    to the end of the larger graph.

    When the pattern is matched, it will be removed from the larger
    function and replaced by ``replacement``. If there are multiple
    matches for ``pattern`` in the larger function, each non-overlapping
    match will be replaced. In the case of a match overlap, the first
    found match in the set of overlapping matches will be replaced.
    ("First" here being defined as the first in a topological ordering
    of the Nodes' use-def relationships. In most cases, the first Node
    is the parameter that appears directly after ``self``, while the
    last Node is whatever the function returns.)

    One important thing to note is that the parameters of the
    ``pattern`` Callable must be used in the Callable itself,
    and the parameters of the ``replacement`` Callable must match
    the pattern. The first rule is why, in the above code block, the
    ``forward`` function has parameters ``x, w1, w2``, but the
    ``pattern`` function only has parameters ``w1, w2``. ``pattern``
    doesn't use ``x``, so it shouldn't specify ``x`` as a parameter.
    As an example of the second rule, consider replacing

    .. code-block:: python

        def pattern(x, y):
            return torch.neg(x) + torch.relu(y)

    with

    .. code-block:: python

        def replacement(x, y):
            return torch.relu(x)

    In this case, ``replacement`` needs the same number of parameters
    as ``pattern`` (both ``x`` and ``y``), even though the parameter
    ``y`` isn't used in ``replacement``.

    After calling ``subgraph_rewriter.replace_pattern``, the generated
    Python code looks like this:

    .. code-block:: python

        def forward(self, x, w1, w2):
            stack_1 = torch.stack([w1, w2])
            sum_1 = stack_1.sum()
            stack_2 = torch.stack([w1, w2])
            sum_2 = stack_2.sum()
            max_1 = torch.max(sum_1)
            add_1 = x + max_1
            max_2 = torch.max(sum_2)
            add_2 = add_1 + max_2
            return add_2
    """
    match_and_replacements = _replace_pattern(gm, pattern, replacement)
    return [Match(anchor=m.anchor, nodes_map=m.nodes_map) for m in match_and_replacements]


# Experimental API, not backward compatible
@compatibility(is_backward_compatible=False)
def replace_pattern_with_filters(
    gm: GraphModule,
    pattern: Union[Callable, Graph, GraphModule],
    replacement: Union[Callable, Graph, GraphModule],
    match_filters: Optional[List[Callable[["InternalMatch", Graph, Graph], bool]]] = None,  # type: ignore[name-defined]
    ignore_literals: bool = False,
) -> List[ReplacedPatterns]:
    """
    See replace_pattern for documentation. This function is an overload with an additional match_filter argument.

    Args:
        ``match_filters``: A list of functions that take in
            (match: InternalMatch, original_graph: Graph, pattern_graph: Graph) and return a boolean indicating
            whether the match satisfies the condition.
            See matcher_utils.py for definition of InternalMatch.
    """

    return _replace_pattern(gm, pattern, replacement, match_filters, ignore_literals)


def _replace_pattern(
    gm: GraphModule,
    pattern: Union[Callable, Graph, GraphModule],
    replacement: Union[Callable, Graph, GraphModule],
    match_filters: Optional[List[Callable[["InternalMatch", Graph, Graph], bool]]] = None,  # type: ignore[name-defined]
    ignore_literals: bool = False,
) -> List[ReplacedPatterns]:
    if match_filters is None:
        match_filters = []

    # Get the graphs for `gm`, `pattern`, `replacement`
    original_graph: Graph = gm.graph

    if isinstance(pattern, GraphModule):
        pattern_graph = pattern.graph
    elif isinstance(pattern, Graph):
        pattern_graph = pattern
    else:
        pattern_graph = symbolic_trace(pattern).graph

    if isinstance(replacement, GraphModule):
        replacement_graph = replacement.graph
    elif isinstance(replacement, Graph):
        replacement_graph = replacement
    else:
        replacement_graph = symbolic_trace(replacement).graph

    matcher = SubgraphMatcher(pattern_graph, match_output=False, match_placeholder=False,
                              remove_overlapping_matches=True, ignore_literals=ignore_literals)
    _matches: List[InternalMatch] = matcher.match(original_graph)

    # Filter out matches that don't match the filter
    _matches = [
        m for m in _matches
        if all(match_filter(m, original_graph, pattern_graph)
               for match_filter in match_filters)
    ]

    replacement_placeholders = [n for n in replacement_graph.nodes if n.op == "placeholder"]

    # As we progressively replace nodes, we'll need to keep track of how the match results should change
    match_changed_node: Dict[Node, Node] = {}

    match_and_replacements = []
    for match in _matches:
        for node in replacement_graph.nodes:
            if node.op == "get_attr" or node.op == "call_module":
                submodule_name = node.target
                if node.op == "get_attr":
                    _, _, submodule_name = node.target.rpartition(".")
                if submodule_name in match.modules_map:
                    to_replace = match.modules_map[submodule_name]
                    node.target = node.target.replace(submodule_name, to_replace, 1)
                    
        # Build connecting between replacement graph's input and original graph input producer node

        # Initialize `val_map` with mappings from placeholder nodes in
        # `replacement` to their corresponding node in `original_graph`
        assert len(match.placeholder_nodes) == len(replacement_placeholders)
        val_map: Dict[Node, Node] = {}
        for rn, gn in zip(replacement_placeholders, match.placeholder_nodes):
            if isinstance(gn, Node):
                val_map[rn] = match_changed_node.get(gn, gn)
                if gn != val_map[rn]:
                    # Update match.placeholder_nodes and match.nodes_map with the node that replaced gn
                    gn_ind = match.placeholder_nodes.index(gn)
                    match.placeholder_nodes[gn_ind] = match_changed_node[gn]
                    map_key = list(match.nodes_map.keys())[list(match.nodes_map.values()).index(gn)]
                    match.nodes_map[map_key] = match_changed_node[gn]
            else:
                val_map[rn] = gn

        # Copy the replacement graph over
        user_nodes: Set[Node] = set()
        for n in match.returning_nodes:
            for user in n.users:
                user_nodes.add(user)
        assert user_nodes, "The returning_nodes should have at least one user node"

        if len(user_nodes) == 1:
            first_user_node = next(iter(user_nodes))
        else:
            # If there are multiple user nodes, we need to find the first user node
            # in the current execution order of the `original_graph`
            for n in original_graph.nodes:
                if n in user_nodes:
                    first_user_node = n
                    break

        with original_graph.inserting_before(first_user_node):
            copied_returning_nodes = original_graph.graph_copy(replacement_graph, val_map)

        if isinstance(copied_returning_nodes, Node):
            copied_returning_nodes = (copied_returning_nodes, )

        # Get a list of nodes that have been replaced into the graph
        replacement_nodes: List[Node] = [v for v in val_map.values() if v not in match.placeholder_nodes]

        # Hook the output Node of the replacement subgraph in to the
        # original Graph at the correct location
        assert len(match.returning_nodes) == len(copied_returning_nodes)
        for gn, copied_node in zip(match.returning_nodes, copied_returning_nodes):
            gn.replace_all_uses_with(copied_node)
            match_changed_node[gn] = copied_node
        # Remove the original nodes
        for node in reversed(pattern_graph.nodes):
            if node.op != "placeholder" and node.op != "output":
                gn = match.nodes_map[node]
                gm.graph.erase_node(gn)

        match_and_replacements.append(
            ReplacedPatterns(
                anchor=match.anchors[0],
                nodes_map=match.nodes_map,
                replacements=replacement_nodes
            )
        )

    # Update the passed-in GraphModule to reflect the new state of
    # `original_graph`
    gm.recompile()

    # If `replacement` was an nn.Module, we'll need to make sure that
    # all the submodules have been copied over correctly
    if isinstance(replacement, torch.nn.Module):
        _replace_attributes(gm, replacement)

    return match_and_replacements
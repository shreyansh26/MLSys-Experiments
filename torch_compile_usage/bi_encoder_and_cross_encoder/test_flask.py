import unittest
import requests

class TestFlaskServer(unittest.TestCase):
    def test_predict(self):
        # Trigger model compilation by calling "/"
        for _ in range(40):
            url = "http://127.0.0.1:8080/predict"
            payload = {
                "sentences": [
                    "This is a test sentence.",
                    "Another sentence for testing.",
                    "Another sentence for testing 3.",
                    "Another sentence for testing 4."
                ] * 1024
            }
            response = requests.post(url, json=payload)
            self.assertEqual(response.status_code, 200, f"Unexpected status code: {response.status_code}")
            data = response.json()
            self.assertIn("inference_time", data, "inference_time missing in response")
            print(data['inference_time'])
            print(len(data['embeddings']))
            self.assertIn("embeddings", data, "embeddings missing in response")
            self.assertIsInstance(data["embeddings"], list)
            # Check that the number of embedding vectors matches the number of sentences sent
            self.assertEqual(len(data["embeddings"]), len(payload["sentences"]))
            # Optionally, check shape of the first embedding vector if available
            if data["embeddings"]:
                self.assertIsInstance(data["embeddings"][0], list)
                self.assertGreater(len(data["embeddings"][0]), 0, "Empty embedding vector returned")

if __name__ == "__main__":
    # root_response = requests.get("http://127.0.0.1:8080/")
    # print(root_response.status_code)
    unittest.main()
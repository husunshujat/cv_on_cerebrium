import requests
import argparse
import base64
import os
import json
import subprocess
import sys
# ===== CONFIGURE THESE =====
API_URL = "https://api.cortex.cerebrium.ai/v4/p-6bedb608/mtailorround2/run"  # replace with your endpoint
BEARER_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTZiZWRiNjA4IiwibmFtZSI6IiIsImRlc2NyaXB0aW9uIjoiIiwiZXhwIjoyMDY0ODY2MjI0fQ.vm-d7phpuzcprsFEUcynH2gCqNjc8rWNaSctMiBZKzVhmCbzGOjLgx4ACeM-5-vlZaq-D74pBBljQTQbfuM5PcOOPNzvtB7EbfMnAsK9xiMJKmZppy3XUYFP2q4X0D2STTEhHNrtBs5ZJsTLNBI6J-c-8KmW_xzayxzgdG7BNM1MoKYbC_OVbHsOfnYi5NpuYu042LnS0lL5y-lgWJXQRbhuNZXrOWIrXHAT_I1Emwvua1tdaTJi5uC8rpQxlrDbmqevZJMs2thRkAJT3uEqV08CmonY2_aPOhmqWJ65kMXX-R_p20DS0kVbkBb6YI3iisFDeO5xONBOlEDcxy5kGw"  # replace with your token
HEADERS = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Content-Type": "application/json"
}

def encode_image_to_base64(image_path):
    """Convert image to base64 string for JSON transmission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def send_image(image_path):
    """Send image to Cerebrium endpoint and print prediction."""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None

    payload = {"image_file_path": image_path}  # match this with how your Cerebrium app expects input

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(result)
            print(f"✅ Prediction for {image_path}: Class ID = {result.get('result').get('predicted_class')}")
            return result.get('result').get('predicted_class')
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")

def run_custom_tests():
    from custom_test import test_images, test_labels
    for i in range(len(test_images)):
        img_pred = send_image(test_images[i])
        if img_pred==test_labels[i]:
            print(f"Case {test_images[i]}: ✅ Successfully passed.")
        else:
            (f"Case {test_images[i]}: ❌ Failed to predict correctly.")

def is_endpoint_alive(image_path: str) -> bool:
    """Returns True if the Cerebrium model responds successfully."""
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False

    payload = {"image_file_path": image_path}

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            print("✅ Endpoint is alive.")
            return True
        else:
            print(f"⚠️ Endpoint responded with status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return False



# Example usage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Cerebrium deployed model")
    parser.add_argument("--image", type=str, help="Path to image to test")
    parser.add_argument("--customtest", action="store_true", help="Run preset custom tests")
    parser.add_argument("--health",type=str, help="Run health check on endpoint")

    args = parser.parse_args()

    if args.health:
        is_endpoint_alive(args.health)
    elif args.test:
        run_custom_tests()
    elif args.image:
        send_image(args.image)
    else:
        print("❗ Provide either --image <path>, --test or --health <path>")

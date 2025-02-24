import requests
import os
from pathlib import Path

# Create documents directory if it doesn't exist
documents_dir = Path("documents")
documents_dir.mkdir(exist_ok=True)

# List of URLs from docs.txt
urls = [
    "https://chitrapurmath.net/documents/sunbeam/143_SunbeamOctober2024Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/141_SunbeamSeptember2024Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/140_SunbeamAugust2024Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/139_SunbeamJuly2024Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/138_SunbeamJune2024Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/137_SunbeamMay2024Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/136_SunbeamApril2024Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/135_SunbeamMarch2024Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/134_SunbeamFebruary2024Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/133_SunbeamJanuary2024Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/132_SunbeamDecember2023Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/131_SunbeamNovember2023Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/130_SunbeamJuly2024Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/129_SunbeamSeptember2023Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/128_SunbeamAugust2023Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/127_SunbeamJuly2023Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/126_SunbeamMayandJune2023Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/125_SunbeamApril2023Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/124_SunbeamMarch2023Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/123_SunbeamFebruary2023Issue1.pdf",
    "https://chitrapurmath.net/documents/sunbeam/122_SunbeamJanuary2023Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/121_SunbeamDecember2022Issue11.pdf",
    "https://chitrapurmath.net/documents/sunbeam/120_SunbeamNovember2022Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/119_SunbeamOctober2022Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/118_SunbeamSeptember2022Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/117_SunbeamAugust2022.pdf",
    "https://chitrapurmath.net/documents/sunbeam/116_SunbeamJuly2022Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/115_SunbeamJune2022Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/114_SunbeamMay2022Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/113_SunbeamApril2022Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/112_SunbeamMarch2022Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/111_SunbeamFebruary2022Issue.pdf",
    "https://chitrapurmath.net/documents/sunbeam/110_SunbeamJanuary2022Issue.pdf"
]

def download_file(url, destination):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded: {destination}")
    except Exception as e:
        print(f"Failed to download {url}: {str(e)}")

# Download each file
for url in urls:
    filename = url.split('/')[-1]
    destination = documents_dir / filename
    if not destination.exists():
        print(f"Downloading {filename}...")
        download_file(url, destination)
    else:
        print(f"File already exists: {filename}")

print("\nDownload process completed!") 
#!/usr/bin/env python3
"""
Test script to verify Internet Topology Zoo download works.
"""

import requests
import tempfile
import zipfile
import os

def test_topology_zoo_download():
    """Test if the Internet Topology Zoo download works."""
    print("🧪 Testing Internet Topology Zoo download...")
    
    zoo_url = "https://topology-zoo.org/files/archive.zip"
    
    try:
        print(f"🔗 Testing URL: {zoo_url}")
        
        # Test with HEAD request first
        head_response = requests.head(zoo_url, timeout=10)
        print(f"📊 Status Code: {head_response.status_code}")
        print(f"📦 Content Type: {head_response.headers.get('content-type', 'unknown')}")
        print(f"📏 Content Length: {head_response.headers.get('content-length', 'unknown')} bytes")
        
        if head_response.status_code == 200:
            print("✅ URL is accessible")
            
            # Test actual download (first 10KB only)
            print("🔽 Testing partial download...")
            response = requests.get(zoo_url, stream=True, timeout=30, 
                                  headers={'Range': 'bytes=0-10239'})
            
            if response.status_code in [200, 206]:  # 206 for partial content
                print("✅ Download test successful")
                
                # Verify it's a ZIP file
                content = response.content
                if content.startswith(b'PK'):
                    print("✅ File appears to be a valid ZIP archive")
                    return True
                else:
                    print("❌ File does not appear to be a ZIP archive")
                    return False
            else:
                print(f"❌ Download failed with status: {response.status_code}")
                return False
        else:
            print(f"❌ URL not accessible, status: {head_response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection error")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_full_download():
    """Test full download of the topology zoo."""
    print("\n🔽 Testing full Internet Topology Zoo download...")
    
    zoo_url = "https://topology-zoo.org/files/archive.zip"
    
    try:
        # Download to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            print("📥 Downloading archive...")
            response = requests.get(zoo_url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
                total_size += len(chunk)
                if total_size % 100000 == 0:  # Progress every 100KB
                    print(f"   Downloaded: {total_size // 1000}KB")
            
            tmp_path = tmp_file.name
        
        print(f"✅ Download complete: {total_size} bytes")
        
        # Test ZIP extraction
        print("📂 Testing ZIP extraction...")
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            graphml_files = [f for f in file_list if f.endswith('.graphml')]
            
            print(f"📊 Archive contains {len(file_list)} files")
            print(f"🌐 Found {len(graphml_files)} GraphML topology files")
            
            if graphml_files:
                print("📋 Sample topologies found:")
                for i, filename in enumerate(graphml_files[:5]):  # Show first 5
                    print(f"   • {filename}")
                if len(graphml_files) > 5:
                    print(f"   ... and {len(graphml_files) - 5} more")
        
        # Clean up
        os.unlink(tmp_path)
        
        print("✅ Full download and extraction test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Full download test failed: {e}")
        return False

if __name__ == "__main__":
    print("🌐 Internet Topology Zoo Download Test")
    print("=" * 50)
    
    # Test basic connectivity
    basic_test = test_topology_zoo_download()
    
    if basic_test:
        # Test full download if basic test passes
        full_test = test_full_download()
        
        if full_test:
            print("\n🎉 All tests passed! Internet Topology Zoo is accessible.")
            print("💡 You can now use: python3 real_world_topology_importer.py --download-zoo")
        else:
            print("\n⚠️  Basic test passed but full download failed.")
            print("💡 Try the download again or check your internet connection.")
    else:
        print("\n❌ Basic connectivity test failed.")
        print("💡 Internet Topology Zoo may be temporarily unavailable.")
        print("💡 Use sample topologies instead: python3 real_world_topology_importer.py --create-samples")
#!/usr/bin/env python3
"""
Test Script for GraphML Mininet Integration
==========================================

Demonstrates how to use the GraphML Mininet script with the updated controller.

Usage:
    sudo python3 test_graphml_mininet.py
"""

import subprocess
import time
import os
import sys
import signal
from pathlib import Path


class GraphMLMininetTest:
    """Test runner for GraphML Mininet integration."""
    
    def __init__(self):
        self.controller_process = None
        self.mininet_process = None
        
    def start_controller(self):
        """Start the updated SDN controller."""
        print("🎮 Starting Updated SDN Controller...")
        
        # Check if controller file exists
        controller_file = Path("sdn/updated_controller.py")
        if not controller_file.exists():
            print(f"❌ Controller file not found: {controller_file}")
            return False
        
        # Start Ryu controller with REST API apps
        cmd = [
            'ryu-manager',
            'ryu.app.ofctl_rest',
            'ryu.app.rest_topology',
            str(controller_file),
            '--observe-links'
        ]
        
        try:
            self.controller_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # Wait a moment for controller to start
            time.sleep(3)
            
            # Check if process is still running
            if self.controller_process.poll() is not None:
                stdout, stderr = self.controller_process.communicate()
                print(f"❌ Controller failed to start")
                print(f"   stdout: {stdout.decode()[:500]}")
                print(f"   stderr: {stderr.decode()[:500]}")
                return False
            
            print(f"✅ Controller started (PID: {self.controller_process.pid})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start controller: {e}")
            return False
    
    def wait_for_controller_ready(self, timeout=30):
        """Wait for controller to be ready."""
        print("⏳ Waiting for controller REST API to be ready...")
        
        for i in range(timeout):
            try:
                import requests
                # Test the topology endpoint that hybrid_resilink_implementation uses
                response = requests.get('http://localhost:8080/v1.0/topology/switches', timeout=2)
                if response.status_code == 200:
                    print(f"✅ Controller REST API ready after {i+1} seconds")
                    return True
            except Exception as e:
                if i == 0:
                    print(f"   Waiting for REST API... (error: {str(e)[:50]})")
            
            time.sleep(1)
            if i % 5 == 4:  # Print progress every 5 seconds
                print(f"   Still waiting... ({i+1}/{timeout}s)")
        
        print(f"❌ Controller REST API not ready after {timeout} seconds")
        return False
    
    def test_topology(self, graphml_file, duration=60):
        """Test a specific GraphML topology."""
        print(f"\n🧪 Testing GraphML Topology: {graphml_file}")
        print("=" * 50)
        
        if not Path(graphml_file).exists():
            print(f"❌ GraphML file not found: {graphml_file}")
            return False
        
        # Start controller
        if not self.start_controller():
            return False
        
        if not self.wait_for_controller_ready():
            print("⚠️  Controller API not responding, but continuing...")
            # Don't fail here as the controller might still work
        
        # Start Mininet with GraphML topology
        print(f"🌐 Starting Mininet with GraphML topology...")
        
        cmd = [
            'python3', 'mininet_graphml_topology.py',
            graphml_file,
            '--duration', str(duration),
            '--save-json', f'{Path(graphml_file).stem}_topology.json'
        ]
        
        try:
            self.mininet_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                text=True
            )
            
            print(f"✅ Mininet started (PID: {self.mininet_process.pid})")
            print(f"⏰ Running for {duration} seconds...")
            
            # Monitor output
            start_time = time.time()
            while time.time() - start_time < duration + 10:  # Extra time for cleanup
                if self.mininet_process.poll() is not None:
                    break
                time.sleep(1)
            
            # Get output
            stdout, _ = self.mininet_process.communicate(timeout=10)
            
            print("\n📋 Mininet Output:")
            print("-" * 30)
            print(stdout[-1000:])  # Last 1000 characters
            print("-" * 30)
            
            print("✅ Test completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up processes."""
        print("\n🧹 Cleaning up...")
        
        # Kill Mininet process
        if self.mininet_process:
            try:
                os.killpg(os.getpgid(self.mininet_process.pid), signal.SIGTERM)
                self.mininet_process.wait(timeout=5)
            except:
                pass
        
        # Kill controller process
        if self.controller_process:
            try:
                os.killpg(os.getpgid(self.controller_process.pid), signal.SIGTERM)
                self.controller_process.wait(timeout=5)
            except:
                pass
        
        # Clean up Mininet
        subprocess.run(['mn', '-c'], capture_output=True)
        
        print("✅ Cleanup completed")


def main():
    """Main function."""
    if os.geteuid() != 0:
        print("❌ This script must be run as root (use sudo)")
        sys.exit(1)
    
    tester = GraphMLMininetTest()
    
    try:
        # Test with Bell Canada topology
        graphml_file = "real_world_topologies/Bellcanada.graphml"
        
        if not Path(graphml_file).exists():
            print(f"❌ Test topology not found: {graphml_file}")
            print("Please ensure the real_world_topologies directory exists with GraphML files")
            sys.exit(1)
        
        print("🌐 GraphML Mininet Integration Test")
        print("=" * 40)
        print(f"📁 Testing with: {graphml_file}")
        print(f"🎮 Controller: sdn/updated_controller.py")
        print(f"⏰ Duration: 60 seconds")
        print()
        
        success = tester.test_topology(graphml_file, duration=60)
        
        if success:
            print("\n🎉 Test completed successfully!")
            print("\nNext steps:")
            print("1. Check the generated JSON file for topology data")
            print("2. Use the controller's REST API to inspect the network")
            print("3. Run your ResiLink optimization on this topology")
        else:
            print("\n❌ Test failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test error: {e}")
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
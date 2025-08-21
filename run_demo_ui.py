import os
import sys
from pathlib import Path

def main():
    print("Smart Voice Command System - Demo UI Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    ui_path = current_dir / "ui" / "voice_system_gui.py"
    
    if not ui_path.exists():
        print("Error: Could not find ui/voice_system_gui.py")
        print("Please make sure you're running this from the project root directory.")
        print(f"Current directory: {current_dir}")
        return 1
    
    # Check for required dependencies
    try:
        import tkinter
        print("tkinter available")
    except ImportError:
        print("Error: tkinter not available")
        print("Please install tkinter (usually comes with Python)")
        return 1
    
    try:
        # Add current directory to Python path
        sys.path.insert(0, str(current_dir))
        
        # Import and run the GUI
        from ui.voice_system_gui import main as gui_main
        
        print("Starting Demo UI...")
        print("\nDemo UI Features:")
        print("• Start/Stop voice recognition")
        print("• Add new voice commands")
        print("• Pause system for safe demonstrations")
        print("• View system status and activity log")
        print("• No database required!")
        print("\n" + "=" * 50)
        
        gui_main()
        
    except ImportError as e:
        print(f"Error importing GUI: {e}")
        print("Make sure all required Python modules are installed.")
        return 1
    except Exception as e:
        print(f"Error starting GUI: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

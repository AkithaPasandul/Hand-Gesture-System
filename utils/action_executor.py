import pyautogui

class ActionExecutor:
    def __init__(self, mapping):
        self.mapping = mapping
        pyautogui.FAILSAFE = False

    def execute(self, gesture):
        action = self.mapping.get(gesture)

        if action == "VOLUME_UP":
            pyautogui.press("volumeup")

        elif action == "VOLUME_MUTE":
            pyautogui.press("volumemute")

        elif action == "NEXT_SLIDE":
            pyautogui.press("right")

        elif action == "NONE":
            pass

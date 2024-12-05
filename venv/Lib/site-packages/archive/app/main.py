
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


from archive.app.controllers import Controller


if __name__ == '__main__':
    app = Controller(None)
    app.view.mainloop()

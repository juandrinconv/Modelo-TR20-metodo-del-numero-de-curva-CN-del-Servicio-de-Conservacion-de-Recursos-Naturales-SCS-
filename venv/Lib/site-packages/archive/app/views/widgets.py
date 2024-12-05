
import tkinter as tk
from tkinter import ttk


class PropertyFrame(tk.Frame):

    def __init__(self, master, text):
        """A view widget for display basic properties.

        Args:
            master (tk.Frame | tk.LabelFrame): The wrapper that houses the widget.
            text (str): The name of the property, that is displayed to the user.

        """
        super(PropertyFrame, self).__init__(master)
        label_frame = tk.LabelFrame(self, text=text, padx=5, pady=5, borderwidth=0)
        label_frame.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES)

        self.entry = tk.Entry(label_frame)
        self.entry.pack(side=tk.LEFT)

    def clear(self):
        """Clear the options of the widget."""
        self.entry.delete(0, tk.END)
        return

    def set(self, text):
        """Replace the displayed text of the widget.

        Args:
            text (str): The new text to display.

        """
        self.clear()
        if text:
            self.entry.insert(0, text)


class CheckFrame(tk.Frame):

    def __init__(self, master, text):
        """A view widget for display boolean properties.

        Args:
            master (tk.Frame | tk.LabelFrame): The wrapper that houses the widget.
            text (str): The name of the property, that is displayed to the user.

        """
        super(CheckFrame, self).__init__(master)
        frame = tk.Frame(self, padx=5, pady=5)
        frame.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES)
        self.value = tk.BooleanVar()
        self.check = tk.Checkbutton(frame, text=text, variable=self.value, onvalue=True, offvalue=False)
        self.check.pack(side=tk.LEFT)

    def set(self, boolean):
        """Toggle the check of the widget.

        Args:
            boolean (bool): Toggles 'on' if True, 'off' if False.

        """
        self.value.set(boolean)


class SelectFrame(tk.Frame):

    def __init__(self, master, text):
        """A view widget for display boolean properties.

        Args:
            master (tk.Frame | tk.LabelFrame): The wrapper that houses the widget.
            text (str): The name of the property, that is displayed to the user.

        """
        super(SelectFrame, self).__init__(master)
        label_frame = tk.LabelFrame(self, text=text, padx=5, pady=5, borderwidth=0)
        label_frame.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES)
        self.value = ""
        self.cb = ttk.Combobox(label_frame, textvariable=self.value)
        self.cb.pack(side=tk.LEFT)

    def set(self, text):
        """Replace the displayed text of the widget.

        Args:
            text (str): The new text to display.

        """
        self.cb.set(text)

    def options(self, texts):
        """Set the possible options.

        Args:
            texts (list[str]): Available options that can be selected.

        """
        self.cb['values'] = sorted(texts)


class CustomListBox(tk.Listbox):

    def __init__(self, master):
        """A subclass of :class:`tk.Listbox` that adds some useful methods.

        Args:
            master (tk.Frame | tk.LabelFrame): The wrapper that houses the widget.

        """
        super(CustomListBox, self).__init__(master, exportselection=False)

    def clear(self):
        """Clear the options of the widget."""
        self.delete(0, tk.END)

    def options(self, texts):
        """Set the possible options.

        Args:
            texts (list[str]): Available options that can be selected.

        """
        self.clear()
        if texts:
            for text in sorted(texts):
                self.insert(tk.END, text)

    def set(self, text):
        """Replace the selected text of the widget.

        Args:
            text (str): The new text to selected.

        Raises:
            ValueError: If text does not match any listbox values.

        """
        for i, item in enumerate(self.get(0, tk.END)):
            if item == text.strip():
                index = i
                self.select_set(index)
                return
        raise ValueError('Text does not match any listbox values')

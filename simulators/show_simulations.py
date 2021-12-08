from IPython.display import Video


def show_simulation(name):
    from pyvirtualdisplay import Display

    # To make the rendering possible
    display = Display(visible=0, size=(1400, 900))
    display.start()

    return Video(f"videos/{name}.mp4")

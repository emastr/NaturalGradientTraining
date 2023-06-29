import time
import matplotlib.pyplot as plt
from natgrad.plotting import barplot, remove_top_right_axes


class EventTracker:

    def __init__(self):
        """

        """
        self.events = {}
        
    def __getitem__(self, key):
        return self.events[key]

    def start_event(self, event_name, verbose=False):
        if event_name not in self.events.keys():
            self.events[event_name] = Event(event_name)
        self.events[event_name].start()
        if verbose:
            print(f"Starting event: {self.events[event_name].to_string()}", end="\r")

    def end_event(self, event_name):
        assert event_name in self.events.keys() , f"Event: {event_name} does not exist in the logger."
        self.events[event_name].end()

    def print_logs(self, file=None):
        if file == None:
            for event in self.events.keys():
                print(self.events[event].to_string())
        else:
            None

    #function to plot a horizontal bar chart with event names and time values
    def barplot_logs(self, ax=None):
        if ax is None:
            ax = plt.gca()
        names = list(self.events.keys())
        ticks = list(range(len(names)))
        tvals = []
        for event in self.events.keys():
            tvals.append(self.events[event].time_total)
        # plot the bars using barplot function with appropriate title and labels
        barplot(ax, names, tvals, title="Event Times", xlabel="Time (s)", ylabel="Event")
        return None
    
    
    def __str__(self):
        string = ""
        for key in self.events.keys():
            string += self.events[key].to_string()
        return string


class Event:
    def __init__(self, name):
        """

        """
        self.calls = 0
        self.time_total = 0
        self.time_mean = 0
        self.name = name
        #
        self.is_running = False
        self.start_time = 0

    def start(self):
        self.is_running = True
        self.start_time = time.time()

    def end(self):
        time_diff = time.time() - self.start_time
        self.start_time = 0

        self.calls += 1
        self.time_total += time_diff
        self.time_mean += (time_diff - self.time_mean) / self.calls
        self.is_running = False


    def to_string(self):
        return f"Event: {self.name}, calls: {self.calls}, time: {self.time_total} s, mean time:{self.time_mean} s."



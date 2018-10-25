import threading
import random
import multiprocessing

def Splitter(words):
    """Parameters:
        words - a string containing a sentance
    Takes as input a sentance, randomly shuffles
    the words in that sentance, then prints the 
    new sentance"""
    mylist = words.split()
    newList = []
    while (mylist):
        newList.append(mylist.pop(random.randrange(0, len(mylist))))
    print(" ".join(newList))
if __name__ == "__main__":
    sentance = "I am a handsom beast. Word."
    numThreads = 5
    threadList = []

    print("Starting \n")
    for i in range(numThreads):
        t = threading.Thread(target=Splitter,
                             args=(sentance,))
        t.start()
        threadList.append(t)

    print("\nThread Count: " + str(threading.activeCount()))
    print("Exiting...\n")
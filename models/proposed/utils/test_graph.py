from graph import Graph
import json

if __name__ == '__main__':
    with open('data/skeletons/openpose.json','r') as f:
        graph = json.load(f)

    adj = Graph(**graph)

    print(adj.A)
    
public static void main(String[] args) {
    long long source = 0, sink = 0, lines;
    long long p, q, k = 1000;
    String filePath;

    long startTime;
    long long totalNodes = 0, totalEdges = 0;
    long long [] adjCount = null, [] adjList = null;
    System.out.print("Choose dataset: ");
    // TODO: input
    rmp = readGraph(filePath, adjList, adjCount, totalNodes, totalEdges, mp);
    System.out.print("Choose input file: ");
    // TODO: input
    // TODO: open input file to read
    System.out.print("How many lines: ");
    // TODO: input

    System.out.println("========= NEW RUN");
    // TODO: format
    System.out.println("This graph contains %lld nodes connected by %lld edges\n");

    float alpha, probability, beta, pmax;
    float kmax, dif;
    long long startNode, outdegree, nextNode, counter = 0;
    boolean flag = true;
    // TODO: random seed
    // TODO: vector<_HyperEdge> hyperEdge, E;
    // TODO: set<LL> nodeSet;
    // TODO: open output file to write
    
}
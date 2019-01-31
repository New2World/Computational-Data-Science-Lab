import java.io.*;
import java.util.*;
import java.lang.Math;
import datastruct.DataStruct.*;

public class Main{
    private static int totalNodes = 0;
    private static int totalEdges = 0;

    private final static int MAXEDGE = 30000000;
    private final static int MAXVERTEX = 15000000;

    private static void readGraph(String filePath) throws Exception{
        Vector<Edge> edge = new Vector<Edge>();
        String text;
        File file = null;
        BufferedReader reader = null;
        file = new File(filePath);
        reader = new BufferedReader(new FileReader(file));
        int u, v, tmp;
        while((text = reader.readLine()) != null){
            String [] textStr = text.split(" ");
            u = Integer.parseInt(textStr[1]);
            v = Integer.parseInt(textStr[0]);
            if(!Globals.mp.containsKey(u)){
                Globals.mp.put(u, ++totalNodes);
                Globals.rmp.put(totalNodes, u);
            }
            if(!Globals.mp.containsKey(v)){
                Globals.mp.put(v, ++totalNodes);
                Globals.rmp.put(totalNodes, v);
            }
            edge.add(new Edge(Globals.mp.get(u), Globals.mp.get(v)));
            totalEdges++;
        }
        reader.close();
        Globals.adjList = new int[totalEdges];
        Globals.adjCount = new int[totalNodes + 1];
        Collections.sort(edge);
        for(int i = 0;i < totalEdges;i++){
            Globals.adjList[i] = edge.elementAt(i).to;
            Globals.adjCount[edge.elementAt(i).from]++;
        }
        for(int i = 1;i <= totalNodes;i++){
            Globals.adjCount[i] += Globals.adjCount[i-1];
        }
    }

    private static int randomNumber(Random rand, Integer range){
        return rand.nextInt(range);
    }

    private static Vector<HyperEdge> MpU(int n_nodes, int n_hedges, int p, int q, Vector<HyperEdge> hyperEdge, Random rand){
        DSH dsh = new DSH();
        int rnd = 0, cnt = 0;
        int threshold = (int)(p - Math.sqrt((double)n_hedges));
        int E_dsize = 0, E_ddsize = 0;
        int sizeRecord = -1;
        Vector<Integer> E_ddash = new Vector<Integer>();
        Set<Integer> E = new HashSet<Integer>();
        Set<Integer> E_dash = new HashSet<Integer>();
        for(int i = 0;i < n_hedges;i++){
            E.add(i + 1);
        }
        while(E_dsize < threshold){
            dsh.buildFlowGraph(n_nodes, E, hyperEdge, q);
            E_ddash = dsh.miniCut();
            E_ddsize = E_ddash.size();
            if(E_dsize + E_ddsize <= p){
                E_dash.addAll(E_ddash);
                E.removeAll(E_ddash);
            }
            else{
                for(int i = 0;i < p - E_dsize;i++){
                    rnd = randomNumber(rand, E_ddsize);
                    E_dash.add(E_ddash.elementAt(rnd));
                    E.remove(E_ddash.elementAt(rnd));
                    E_ddash.remove(rnd);
                }
            }
            E_dsize = E_dash.size();
            if(E_dsize == sizeRecord || E_dsize >= n_hedges){
                System.out.print("ERROR (DEAD) - ");
                break;
            }
            sizeRecord = E_dsize;
        }
        Vector<HyperEdge> cardinality = new Vector<HyperEdge>();
        Vector<HyperEdge> result = new Vector<HyperEdge>();
        for(Integer i: E){
            cardinality.add(hyperEdge.elementAt((int)i-1));
        }
        Collections.sort(cardinality);
        cnt = 0;
        for(HyperEdge e: cardinality){
            result.add(e);
            cnt++;
            if(cnt >= p - E_dsize){
                break;
            }
        }
        for(Integer i: E_dash){
            result.add(hyperEdge.elementAt(i-1));
        }
        return result;
    }

    public static void main(String[] args) throws Exception {
        int lines = 0;
        int source = 0, sink = 0;
        int p, q, k = 1000;
        String filePath, text, outputCache;
        List<String> inputStrArr = new ArrayList<String>();

        int startTime;
        File file = null, outputFile = null;
        BufferedReader reader = null;
        BufferedWriter writer = null;
        Scanner inputReader = new Scanner(System.in);
        System.out.print("Choose dataset: ");
        // filePath = inputReader.nextLine();
        filePath = "../../data/hepph/hepph.txt";
        readGraph(filePath);
        System.out.print("Choose input file: ");
        // filePath = inputReader.nextLine();
        filePath = "../../data/hepph/input.txt";
        file = new File(filePath);
        reader = new BufferedReader(new FileReader(file));
        System.out.print("How many lines: ");
        // lines = inputReader.nextInt();
        lines = 3;

        while((text = reader.readLine()) != null){
            inputStrArr.add(text);
        }
        reader.close();
        if(lines <= 0){
            lines = inputStrArr.size();
        }
        else{
            lines = Math.min(lines, inputStrArr.size());
        }

        System.out.println("========= NEW RUN");
        System.out.printf("This graph contains %d nodes connected by %d edges\n\n", totalNodes, totalEdges);

        float alpha, probability, beta, pmax;
        float kmax, dif;
        int startNode, outdegree, nextNode, counter = 0;
        boolean flag = true;
        Random rand = new Random();
        Vector<HyperEdge> hyperEdge = new Vector<HyperEdge>();
        Vector<HyperEdge> E = new Vector<HyperEdge>();
        Set<Integer> nodeSet = new HashSet<Integer>();
        outputFile = new File("output.txt");
        writer = new BufferedWriter(new FileWriter(file));
        for(int l = 0;l < lines;l++){
            if(inputStrArr.get(l) == null){
                break;
            }
            String [] arrOfStr = inputStrArr.get(l).split(" ");
            sink = Globals.mp.get(Integer.parseInt(arrOfStr[1]));
            source = Globals.mp.get(Integer.parseInt(arrOfStr[3]));
            k = Integer.parseInt(arrOfStr[7]);
            beta = Float.parseFloat(arrOfStr[11]);
            System.out.println(sink);
            for(int i = 0;i < k;i++){
                startNode = source;
                nodeSet.clear();
                nodeSet.add(startNode);
                flag = true;
                while(true){
                    outdegree = Globals.adjCount[startNode] - Globals.adjCount[startNode - 1];
                    probability = 1.f * rand.nextFloat() * outdegree;
                    nextNode = Globals.adjCount[startNode - 1] + (int)Math.floor(probability);
                    if(nextNode >= Globals.adjCount[startNode]){
                        nodeSet.clear();
                        break;
                    }
                    for(int j = Globals.adjCount[startNode-1];j < Globals.adjCount[startNode];j++){
                        if(Globals.adjList[j] == sink){
                            nodeSet.remove((Integer)startNode);
                            flag = false;
                            break;
                        }
                    }
                    if(!flag){
                        break;
                    }
                    startNode = Globals.adjList[nextNode];
                    if(nodeSet.contains(startNode)){
                        nodeSet.clear();
                        break;
                    }
                    nodeSet.add(startNode);
                }
                if(nodeSet.size() == 0){
                    continue;
                }
                hyperEdge.add(new HyperEdge(nodeSet));
            }
            q = ((hyperEdge.size() / totalNodes) + hyperEdge.size()) / 2;
            p = (int)(beta * hyperEdge.size());

            nodeSet.clear();
            E.clear();
            if(hyperEdge.size() > 0){
                E.addAll(MpU(totalNodes, hyperEdge.size(), p, q, hyperEdge, rand));
            }
            for(int i = 0;i < E.size();i++){
                nodeSet.addAll(E.elementAt(i).vertex);
            }
            for(Integer i: nodeSet){
                outputCache = Integer.toString(Globals.rmp.get(i));
                writer.write(outputCache + " ");
                System.out.print(outputCache + " ");
            }
            if(lines > 0 && counter >= lines){
                break;
            }
            System.out.println("");
            writer.write("\n");
        }

        writer.close();

        System.out.println("========= FINISH");
    }
}
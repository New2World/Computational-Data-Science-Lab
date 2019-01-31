package datastruct;

import java.util.*;
import java.lang.ref.SoftReference;

public class DataStruct{
    public static class Globals{
        public static int [] adjCount;
        public static int [] adjList;
        public static Map<Integer, Integer> mp = new HashMap<Integer, Integer>();
        public static Map<Integer, Integer> rmp = new HashMap<Integer, Integer>();
    }

    public static class Edge implements Comparable<Edge>{
        public int from, to;

        public Edge(int u, int v){
            this.from = u;
            this.to = v;
        }

        @Override
        public int compareTo(Edge e) {
            if(this.from == e.from){
                if(this.to < e.to){
                    return -1;
                }else if(this.to > e.to){
                    return 1;
                }
                return 0;
            }
            if(this.from < e.from){
                return -1;
            }else if(this.from > e.from){
                return 1;
            }
            return 0;
        }
    }

    public static class HyperEdge implements Comparable<HyperEdge>{
        public Set<Integer> vertex = new HashSet<Integer>();

        public HyperEdge(Set<Integer> vertexSet){
            this.vertex.addAll(vertexSet);
        }

        @Override
        public int compareTo(HyperEdge e){
            int this_size = this.vertex.size();
            int e_size = e.vertex.size();
            if(this_size < e_size){
                return -1;
            }else if(this_size > e_size){
                return 1;
            }
            return 0;
        }
    }

    public static class DSH{
        private final int MAXEDGE = 30000000;
        private final int MAXVERTEX = 15000000;
        private final int INF = 0x3f3f3f3f;

        private class FlowEdge{
            int from, to, cap, next;
            int index;
            public FlowEdge(int f, int t, int c, int n, int i){
                this.from = f;
                this.to = t;
                this.cap = c;
                this.next = n;
                this.index = i;
            }
        }

        private int [] adjHead;
        private int [] dis;
        private int [] pre;
        private int [] gap;
        private FlowEdge [] flowEdge;

        private int edgeCount, vertexCount;
        private int tolHyperEdge, tolVertex;
        private int superSrc, superSink;

        public DSH(){
            adjHead = new int[MAXVERTEX];
            dis = new int[MAXVERTEX];
            pre = new int[MAXVERTEX];
            gap = new int[MAXVERTEX];
            flowEdge = new FlowEdge[MAXEDGE];
        }

        private void clearAll(){
            edgeCount = vertexCount = 0;
            tolHyperEdge = tolVertex = 0;
            superSrc = superSink = 0;
            Arrays.fill(adjHead, -1);
            Arrays.fill(dis, 0);
            Arrays.fill(pre, -1);
            Arrays.fill(gap, 0);
        }

        private void addFlowEdge(int u, int v, int cap, int index){
            flowEdge[edgeCount] = new FlowEdge(u, v, cap, adjHead[u], index);
            adjHead[u] = edgeCount++;
            flowEdge[edgeCount] = new FlowEdge(v, u, 0, adjHead[v], index);
            adjHead[v] = edgeCount++;
        }

        public void buildFlowGraph(int n_nodes, Set<Integer> edgeSet, Vector<HyperEdge> hyperEdge, int q){
            clearAll();
            tolHyperEdge = edgeSet.size();
            tolVertex = n_nodes;
            int cur = 1, temp = 0;
            for(int i: edgeSet){
                temp += hyperEdge.elementAt(i-1).vertex.size();
                addFlowEdge(0, cur, 1, i);
                cur++;
            }
            cur = 1;
            for(int i: edgeSet){
                for(int j: hyperEdge.elementAt(i-1).vertex){
                    addFlowEdge(cur, j + tolHyperEdge, INF, 0);
                }
                cur++;
            }
            cur += n_nodes;
            superSink = cur;
            vertexCount = superSink + 1;
            for(int i = 1;i <= n_nodes;i++){
                addFlowEdge(i + tolHyperEdge, superSink, q, 0);
            }
        }

        public void maxFlow(){
            int flow = 0;
            int aug = INF;
            int u, v, u_cpy;
            boolean flag;
            gap[0] = vertexCount;
            u = pre[0] = 0;
            int [] cur = Arrays.copyOf(adjHead, MAXVERTEX);
            while(dis[0] < vertexCount){
                flag = false;
                // check if right
                u_cpy = u;
                for(int j = cur[u];j != -1;j = flowEdge[j].next){
                    v = flowEdge[j].to;
                    if(flowEdge[j].cap > 0 && dis[u] == dis[v] + 1){
                        flag = true;
                        if(flowEdge[j].cap < aug){
                            aug = flowEdge[j].cap;
                        }
                        pre[v] = u;
                        u = v;
                        if(u == vertexCount - 1){
                            flow += aug;
                            while(u != 0){
                                u = pre[u];
                                flowEdge[cur[u]].cap -= aug;
                                flowEdge[cur[u] ^ 1].cap += aug;
                            }
                            aug = INF;
                        }
                        break;
                    }
                    cur[u_cpy] = j;
                }
                if(flag){
                    continue;
                }
                int mindis = vertexCount;
                for(int j = adjHead[u];j != -1;j = flowEdge[j].next){
                    v = flowEdge[j].to;
                    if(flowEdge[j].cap > 0 && dis[v] < mindis){
                        mindis = dis[v];
                        cur[u] = j;
                    }
                }
                if((--gap[dis[u]]) == 0){
                    break;
                }
                gap[dis[u] = mindis + 1]++;
                u = pre[u];
            }
        }

        public Vector<Integer> miniCut(){
            maxFlow();
            Vector<Integer> minicut = new Vector<Integer>();
            for(int i = 0;flowEdge[i].from == 0;i += 2){
                if(flowEdge[i].cap <= 0){
                    minicut.add(flowEdge[i].index);
                }
            }
            return minicut;
        }
    }
}

package adaptive_time;


import java.util.HashMap;
import java.util.LinkedList;

public class MySortedMap {

    public LinkedList<Integer> node_order;
    HashMap<Integer, Double> valueList=new HashMap<Integer, Double>();
    int size;

    public MySortedMap()
    {
        node_order=new LinkedList<Integer>();
        valueList=new HashMap<Integer, Double>();
        size=0;
    }

    public void push_back(int node, double value)
    {
        node_order.add(node);
        valueList.put(node,value);
        size++;
    }

    public void remove(int node)
    {
        if(node_order.remove((Integer)node))
        {
            if(valueList.remove(node)==null)
            {
                throw new ArithmeticException("remove fail: node is associated with null ");
            }
        }
        else
        {
            throw new ArithmeticException("remove fail: no node exists");
        }
    }
    /*
    public int insert(double value)
    {
        valueList.add(value);
        for(int i=0;i<node_order.size();i++)
        {
            if(value>= valueList.get(node_order.get(i)))
            {
                node_order.add(i, valueList.size()-1);
                return i;
            }
        }
        node_order.add(valueList.size()-1);
        return node_order.size()-1;
    }*/

    public int update(int node, double value)
    {
        //System.out.println(node_order.size()+"  update is running");

        valueList.put(node,value);

        if(node_order.remove((Integer)node))
        {

            int lower=0;
            int upper=node_order.size();
            return insert(node, value, lower, upper);
        }
        else
        {
            System.out.println(node+" does not in the list");
            return -1;
        }
    }

    public int get(int index)
    {
        if(index>node_order.size())
        {
            throw new ArithmeticException("MysortedMap Get out of bound");
        }
        return node_order.get(index);
    }

    public double getvalue(int node)
    {
        if(!valueList.containsKey(node))
        {
            throw new ArithmeticException("MysortedMap getvalue node not exist");
        }
        return valueList.get(node);
    }

    public boolean contains(int node)
    {
        return node_order.contains(node);
    }
    public int insert(int node, double value)
    {
        if(valueList.containsKey(node))
        {
            throw new ArithmeticException("double insert, should use update");

        }
        valueList.put(node,value);
        if(size==0)
        {
            node_order.add(node);
            size++;
            return 0;
        }
        else
        {
            return insert(node,value,0,size);
        }

    }

    private int insert(int node, double value, int lower, int upper)
    {

        //System.out.println(node+" "+value+" "+lower+" "+upper);
        int mid=-1;
        if(value <= valueList.get(node_order.get(upper-1)))
        {
            node_order.add(upper, node);
            size++;
            return upper;
        }
        else if (value >= valueList.get(node_order.get(lower)))
        {
            node_order.add(lower, node);
            size++;
            return lower;
        }
        else if(upper-lower < 10)
        {
            for(int i=lower; i < upper; i++)
            {
                if(value>= valueList.get(node_order.get(i)))
                {

                    node_order.add(i, node);
                    size++;
                    //System.out.println(node+" "+value+" "+valueList.get(node_order.get(i))+" "+node_order.get(i));
                    return i;
                }
            }
            throw new ArithmeticException("something wrong");
            //node_order.add(i,index);
            //return i;
        }
        else
        {
            mid=(lower+upper)/2;
            if(value>= valueList.get(node_order.get(mid)))
            {
                return insert(node, value, lower, mid);
            }
            else
            {
                return insert(node, value, mid, upper);
            }
        }
        //System.out.println(lower+" "+upper+" "+mid);
        //System.out.println(valueList.get(node_order.get(lower))+" "+valueList.get(node_order.get(upper-1))+" "+value);

        //throw new ArithmeticException("insert may wrong");


    }

    // public static void main(String[] args) {
    //     // TODO Auto-generated method stub
    //     MySortedMap map=new MySortedMap();
    //     for(int i=0;i<10;i++)
    //     {
    //         double temp=Math.random();
    //         System.out.println(i+" "+temp);
    //         System.out.println();
    //         map.insert(i, temp);
    //         for(int j=0;j<=i;j++)
    //         {
    //             System.out.println(map.get(j)+" "+map.getvalue(map.get(j)));
    //         }
    //         System.out.println("---------------------------");
    //     }
    // }

}

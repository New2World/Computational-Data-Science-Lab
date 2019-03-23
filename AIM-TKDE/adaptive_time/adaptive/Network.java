//hashmap, load the relationship in memory
package adaptive;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

public class Network {

	public int vertexNum;
	int edgeNum;
	public ArrayList<ArrayList<Integer>> neighbor;
	public ArrayList<ArrayList<Integer>> neighbor_reverse;
	public ArrayList<ArrayList<Double>> probability;
	//public ArrayList<ArrayList<Double>> probability_reverse;
	public ArrayList<Integer> sort_by_degree;
	public int[] inDegree;
	public int[] outDegree;
	//public int[] inDegree_reverse;
	//public double[] intrinsic;
	//public double price;
	//public double coupon;
	//private String randoms_0_1;
	public double[] threshold;
	public double[] c_threshold;
	public double[] s_contri;
	public String type;
	public String path;
	public double IC_prob;

	public int[] s_contri_order;
	public String s_contri_path;
	public boolean is_s_contri=false;;

	public Network()
	{
		neighbor=new ArrayList<ArrayList<Integer>>();
	}

	public Network(String path, String type , int vertexNum)
	{
		System.out.println("importing "+ path+" "+type);
		this.path=path;
		this.vertexNum=vertexNum;
		//this.price=price;
		//this.coupon=coupon;
		//this.randoms_0_1=randoms_0_1;
		this.type=type;
		neighbor=new ArrayList<ArrayList<Integer>>();
		neighbor_reverse=new ArrayList<ArrayList<Integer>>();
		inDegree=new int[vertexNum];
		outDegree=new int[vertexNum];
		sort_by_degree= new ArrayList<Integer> ();
		probability=new ArrayList<ArrayList<Double>>();
		//probability_reverse=new ArrayList<ArrayList<Double>>();
		//Set_intrinsic(price,coupon,randoms_0_1);
		ImportRelation(path);


		switch(type)
		{
			case "IC":
				IC_prob=0.1;
				break;
			case "WC":
				break;
			case "VIC":
				break;
			case "LT":
				threshold=new double[this.vertexNum];
				c_threshold=new double[this.vertexNum];
				break;
			default:
				System.out.print("Invalid model");
		}

	}

	public void set_ic_prob(double prob)
	{
		this.IC_prob=prob;
	}

	public void sort_by_degree()
	{

		Map<Integer, Integer> tempmap = new HashMap<>();
		for(int i=0;i<vertexNum;i++)
		{
			tempmap.put(i, outDegree[i]);
		}
		//Map<String, Integer> sorted;
		Map<Integer, Integer> sortedMap = tempmap.entrySet().stream()
                .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                .collect(Collectors.toMap(Entry::getKey, Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));
		/*
		MySortedMap map=new MySortedMap();
		for(int i=0;i<vertexNum;i++)
		{
			map.insert(i, outDegree[i]);
			if(i% 5000 ==0)
			{
				System.out.println(i);
			}

		}*/
		for (Map.Entry<Integer, Integer> entry : sortedMap.entrySet())
		{
		    //System.out.println(entry.getKey() + "/" + entry.getValue());
		    sort_by_degree.add(entry.getKey());
		}
		/*
		for(int i=0;i<vertexNum;i++)
		{
			sort_by_degree.add(map.node_order.get(i));
		}*/

	}

	public void set_s_contri(String path)
	{
		s_contri=new double[this.vertexNum];
		s_contri_order=new int[this.vertexNum];
		for (int i=0;i<vertexNum;i++)
		{
			s_contri[i]=0;
			//probability.add(tempp);
			//probability_reverse.add(tempp_reverse);
		}
		int index=0;
		File singleFile=new File(path);
		String inString = "";
		try {
            BufferedReader reader = new BufferedReader(new FileReader(singleFile));
            while((inString = reader.readLine())!= null){
            	//System.out.println(inString);
            	//System.out.println(inString);
            	String[] tempString = null;
    			tempString=inString.split(" ");
    			int node=Integer.parseInt(tempString[0]);
    			double value=Double.parseDouble(tempString[1]);
    			s_contri[node]=value;
    			s_contri_order[index]=node;
    			index++;
    			//probability.get(node_1).add(prob);
    			//probability_reverse.get(node_2).add(prob);
            }
            reader.close();
            is_s_contri=true;
            s_contri_path=path;
        } catch (FileNotFoundException ex) {
            System.out.println(path+" The path of data is incorrect.");
        } catch (IOException ex) {
            System.out.println("Error in reading data.");
        }
	}


	public void ImportRelation(String path)
	{
		//System.out.println("Fix "+prob);
		for (int i=0;i<vertexNum;i++)
		{
			ArrayList<Integer> temp=new ArrayList<Integer>();
			ArrayList<Integer> temp_reverse=new ArrayList<Integer>();
			ArrayList<Double> tempp=new ArrayList<Double>();
			//ArrayList<Double> tempp_reverse=new ArrayList<Double>();
			neighbor.add(temp);
			neighbor_reverse.add(temp_reverse);
			inDegree[i]=0;
			outDegree[i]=0;
			probability.add(tempp);
			//probability_reverse.add(tempp_reverse);
		}

		File singleFile=new File(path);
		String inString = "";
		int node_1, node_2;
		try {
            BufferedReader reader = new BufferedReader(new FileReader(singleFile));
            while((inString = reader.readLine())!= null){
            	//System.out.println(inString);
            	//System.out.println(inString);
            	String[] tempString = null;
    			tempString=inString.split(" ");
    			switch(type)
    			{
    				case "VIC":
    					node_1=Integer.parseInt(tempString[0]);
    	    			node_2=Integer.parseInt(tempString[1]);
    	    			double prob=Double.parseDouble(tempString[2]);

    	    			neighbor.get(node_1).add(node_2);
    	    			neighbor_reverse.get(node_2).add(node_1);
    	    			probability.get(node_1).add(prob);
    	    			//probability_reverse.get(node_2).add(prob);
    	    			inDegree[node_2]++;
    	    			outDegree[node_1]++;
    					break;

    				default:
    					node_1=Integer.parseInt(tempString[0]);
    	    			node_2=Integer.parseInt(tempString[1]);
    	    			neighbor.get(node_1).add(node_2);
    	    			neighbor_reverse.get(node_2).add(node_1);
    	    			inDegree[node_2]++;
    	    			outDegree[node_1]++;
    			}

    			//probability.get(node_1).add(prob);
    			//probability_reverse.get(node_2).add(prob);
            }
            reader.close();
        } catch (FileNotFoundException ex) {
            System.out.println(path+" The path of data is incorrect.");
        } catch (IOException ex) {
            System.out.println("Error in reading data.");
        }
	}
	//GET RRSETS FROM RANDOM NODE.
	public void get_rrsets(ArrayList<ArrayList<Integer>> rrsets,int size)
	{
		//ArrayList<ArrayList<Integer>> re_neighbor;

		for(int i=0;i<size;i++)
		{
			ArrayList<Integer> rrset=new ArrayList<Integer>();
			//long startTime = System.currentTimeMillis();

			get_rrset(this.neighbor_reverse,rrset);
			//seed.add((int) Math.round(Math.random()*network.vertexNum));
			//network.spread(seed, 1);

			//long endTime = System.currentTimeMillis();
			//long searchTime = endTime - startTime;
			//System.out.println("time "+searchTime*0.001);
			rrsets.add(rrset);
			if(i % 100000 ==0)
			{
				//System.out.println(i);
			}

		}
		//System.out.println(size+ " rrsets generated.");

	}
	public void get_rrset(ArrayList<ArrayList<Integer>> re_neighbor,ArrayList<Integer> rrset)
	{
		int centerIndex = (int)(Math.floor(Math.random()*vertexNum));
		//long startTime = System.currentTimeMillis();
		//System.out.println("centerIndex "+centerIndex);
		switch(type)
		{
			case "IC":
				re_spreadOnce(re_neighbor,centerIndex,rrset);
				break;
			case "VIC":
				re_spreadOnce(re_neighbor,centerIndex,rrset);
				break;
			case "WC":
				re_spreadOnce(re_neighbor,centerIndex,rrset);
				break;
			case "LT":
				re_spreadOnceLT(re_neighbor,centerIndex,rrset);

				break;
			default:
				System.out.print("Invalid model");
		}


	}






	public void re_spreadOnceLT(ArrayList<ArrayList<Integer>> re_neighbor,int cindex, ArrayList<Integer> rrset)
	{
		rrset.add(cindex);
		while(true)
		{
			if(re_neighbor.get(cindex).size()==0)
			{
				return;
			}
			cindex=re_neighbor.get(cindex).get((int) Math.floor(Math.random()*re_neighbor.get(cindex).size()));
			if(!rrset.contains((Integer)cindex))
			{

				rrset.add(cindex);

			}
			else
			{
				return;
			}

		}
	}

	public void re_spreadOnce(ArrayList<ArrayList<Integer>> re_neighbor,int cindex, ArrayList<Integer> rrset)
	{
		ArrayList<Boolean> state =new ArrayList<Boolean>();
		for(int i=0;i<this.vertexNum;i++)
		{
			state.add(false);
		}
		ArrayList<Integer> newActive =new ArrayList<Integer>();


		state.set(cindex,true);
		rrset.add(cindex);


		while(newActive.size()>0)
		{
			re_spreadOneRound(re_neighbor, newActive, state,rrset);

		}


	}

	public void re_spreadOneRound(ArrayList<ArrayList<Integer>> re_neighbor, ArrayList<Integer> newActive, ArrayList<Boolean> state,ArrayList<Integer> rrset)
	 {
		 	ArrayList<Integer> newActiveTemp=new ArrayList<Integer>();
			//System.out.println("spreadOneRound");
			//int a=0;
			//System.out.println(newActive.size());
			for(int i=0;i<newActive.size();i++)
			{

				int cseed=newActive.get(i);
				ArrayList<Integer> cseed_neighbor=re_neighbor.get(cseed);
				//System.out.println(i+" "+cseed_neighbor.size());
				for(int j=0;j<cseed_neighbor.size();j++)
				{
					//a++;
					int cseede=cseed_neighbor.get(j);

					double prob=get_prob(cseede,cseed);
					//System.out.println(prob);
					//System.out.println(prob);
					//System.out.println(cseed+" "+ prob+" "+cseed_neighbor.size());
					if(isSuccess(prob) && !state.get(cseede))
					{
						rrset.add(cseede);
						state.set(cseede,true);
						//System.out.println("state.set(cseede, -cState); "+ -cState);
						newActiveTemp.add(cseede);
					}
				}


			}
			//System.out.println("a             "+a);
			newActive.clear();
			for(int i=0;i<newActiveTemp.size();i++)
			{
				newActive.add(newActiveTemp.get(i));
			}
	 }
	/*
	 public double get_reprob(int cseed,int cseede)
	{
		switch(this.type)
		{
		case "VIC":

			return probability_reverse.get(cseed),cseede);
		case "IC":
				return IC_prob;
			case "WC":
				return 1/(double) inDegree[cseede];
			case "LT":
				return 0.0;
			default:
				System.out.print("Invalid model");
				return 0.0;
		}
	}*/
	public double spread(ArrayList<Integer> seedSet, int times)
	{
	 	//HashMap<Integer,ArrayList<Integer>> neighbor=new HashMap<Integer,ArrayList<Integer>>();
	 	double result=0;
		//System.out.println("spread");
		for(int i=0;i<times;i++)
		{
			result = result+ spreadOnce(seedSet);
			//System.out.println(i);
		}
		return result/times;
	}

	public double spreadOnce(ArrayList<Integer> seedSet)
	{
		//System.out.println("spreadonce");
		if(this.type.equals("LT"))
		{
			for(int i=0;i<this.vertexNum;i++)
			{
				threshold[i]=Math.random();
				c_threshold[i]=0;
			}
		}
		ArrayList<Boolean> state =new ArrayList<Boolean>();
		for(int i=0;i<this.vertexNum;i++)
		{
			state.add(false);
		}
		ArrayList<Integer> newActive =new ArrayList<Integer>();

		for(int j=0;j<seedSet.size();j++)
		{
			state.set(seedSet.get(j),true);
			//if(intrinsic[seedSet.get(j)] >= price)
			newActive.add(seedSet.get(j));
		}

		while(newActive.size()>0)
		{

			switch(type)
			{
				case "IC":
					spreadOneRound(this.neighbor, newActive, state);
					break;
				case "VIC":
					spreadOneRound(this.neighbor, newActive, state);
					break;
				case "WC":
					spreadOneRound(this.neighbor, newActive, state);
					break;
				case "LT":
					spreadOneRoundLT(this.neighbor, newActive,state);
					break;
				default:
					System.out.print("Invalid model");
			}

		}
		double result=0;
		for(int i=0;i<this.vertexNum;i++)
		{
			if(state.get(i)) //get c_r-active nodes
			{
				result=result+1;
			}
		}
		return result;
	}

	public void spreadOneRoundLT(ArrayList<ArrayList<Integer>> relationship, ArrayList<Integer> newActive, ArrayList<Boolean> state)
	{
		ArrayList<Integer> newActiveTemp=new ArrayList<Integer>();
		for(int i=0;i<newActive.size();i++)
		{

			int cseed=newActive.get(i);

			ArrayList<Integer> cseed_neighbor=relationship.get(cseed);
			for(int j=0;j<cseed_neighbor.size();j++)
			{
				//a++;
				int cseede=cseed_neighbor.get(j);
				if(!state.get(cseede))
				{
					c_threshold[cseede]=c_threshold[cseede]+1/(double) inDegree[cseede];
					if(c_threshold[cseede]>threshold[cseede])
					{
							state.set(cseede, true);
							newActiveTemp.add(cseede);
					}
				}
			}
		}
		newActive.clear();
		for(int i=0;i<newActiveTemp.size();i++)
		{
			newActive.add(newActiveTemp.get(i));
		}
	}

	public void spreadOneRound(ArrayList<ArrayList<Integer>> relationship, ArrayList<Integer> newActive, ArrayList<Boolean> state)
	{
		ArrayList<Integer> newActiveTemp=new ArrayList<Integer>();
		//System.out.println("spreadOneRound");
		//int a=0;
		//System.out.println(newActive.size());
		for(int i=0;i<newActive.size();i++)
		{

			int cseed=newActive.get(i);

			ArrayList<Integer> cseed_neighbor=relationship.get(cseed);
			//System.out.println(i+" "+cseed_neighbor.size());
			for(int j=0;j<cseed_neighbor.size();j++)
			{
				//a++;
				int cseede=cseed_neighbor.get(j);
				double probability=get_prob(cseed,cseede);
				//System.out.println(probability);
				if(isSuccess(probability))
				{
					if(!state.get(cseede))
					{
						state.set(cseede, true);
						newActiveTemp.add(cseede);
					}
				}
			}
		}
		//System.out.println("a             "+a);
		newActive.clear();
		for(int i=0;i<newActiveTemp.size();i++)
		{
			newActive.add(newActiveTemp.get(i));
		}
	}

	public double get_prob(int cseed,int cseede)
	{
		switch(this.type)
		{
			case "IC":
				return IC_prob;
			case "VIC":
				int index=neighbor.get(cseed).indexOf(cseede);
				if(index==-1)
				{
					throw new ArithmeticException("get_prob "+cseed+" "+cseede);
				}
				return probability.get(cseed).get(index);
			case "WC":
				return 1/(double) inDegree[cseede];
			case "LT":
				return 0.0;
			default:
				System.out.print("Invalid model");
				return 0.0;
		}
	}

	public double get_prob_by_index(int cseed,int cseede)
	{
		switch(this.type)
		{
			case "IC":
				return IC_prob;
			case "VIC":
				int index=neighbor.get(cseed).indexOf(cseede);
				if(index==-1)
				{
					throw new ArithmeticException("get_prob "+cseed+" "+cseede);
				}
				return probability.get(cseed).get(index);
			case "WC":
				return 1/(double) inDegree[cseede];
			case "LT":
				return 0.0;
			default:
				System.out.print("Invalid model");
				return 0.0;
		}
	}

	public void chanageToRelization()
	{
		for(int i=0;i<neighbor.size();i++)
		{
			ArrayList<Integer> temp=new ArrayList<Integer>();
			int cseed=i;
			for(int j=0;j<neighbor.get(i).size();j++)
			{
				int cseede=neighbor.get(i).get(j);
				if(isSuccess(get_prob(cseed,cseede)))
				{
					temp.add(cseede);
				}

			}
			for(int j=0;j<temp.size();j++)
			{
				neighbor.get(i).remove((Integer)temp.get(j));

			}
		}
		neighbor_reverse.clear();
	}
	public boolean isSuccess(double prob)
	{
		if(Math.random() < prob)
		{
			return true;
		}
		else
		{
			return false;
		}

	}


	public void ShowData()
	{
		int edgenum=0;
		for(int i=0;i<vertexNum;i++)
		{
			for(int j=0;j<neighbor.get(i).size();j++)
			{
				System.out.println(i+" "+neighbor.get(i).get(j));
				edgenum++;
			}
		}
		System.out.print(edgenum);
	}

	public void show_s_contri()
	{
		for(int i=0;i<vertexNum;i++)
		{
			System.out.println(i+" "+s_contri[i]);
		}
	}
	//public double averageDegree()
	//{
	//
	//}
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		//String name="dblp";
		//int vnum=430000;
		//String type="WC";

		String name="hepph";
		int vnum=35000;
		String type="WC";

		String path="data/"+name+".txt";
		Network network=new Network(path, type , vnum);
		
		network.sort_by_degree();

		System.out.println("-------------");
		for(int i=0;i<network.vertexNum;i++)
    	{
    		System.out.println(network.outDegree[network.sort_by_degree.get(i)]);

    	}



	}

}

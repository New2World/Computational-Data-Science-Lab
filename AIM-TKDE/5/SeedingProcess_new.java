package adaptive;


import java.util.ArrayList;

import adaptive.Policy.Command;

public class SeedingProcess_new{

	
	public static void MultiGo(Network network, Command command, int simutimes, int round, int budget)
	{
		//ArrayList<Double> c_result=new ArrayList<Double>();
		System.out.println("MultiGo");
		double result=0;
		for(int i=0; i<simutimes; i++)
		{
			//c_result.clear();
			System.out.println("Simulation number "+i);
			result=result+Go(network, command, round, budget);
		}
		System.out.println(result/simutimes);
	}
	
	public static double Go(Network network, Command command, int round, int budget)
	{
		//System.out.println("Go");
		DiffusionState diffusionState=new DiffusionState(network, round, budget);
		double influence=0;
		for(int i=0; i<round; i++)
		{
			//System.out.println("Round "+i+"-"+round);
			ArrayList<Integer> seed_set=new ArrayList<Integer>();
			seed_set=command.compute_seed_set(network, diffusionState);
			//Tools.printlistln(seed_set);
			//System.out.println("seed set size "+seed_set.size());
			diffusionState.seed(seed_set);
			influence=diffusionState.diffuse(network, 1);
		}
		//System.out.println();
		return influence;
		
	}

	
	/*
	private static void spreadOneRound(Network network, DiffusionState diffusionState) 
	{
		// TODO Auto-generated method stub
		ArrayList<Integer> newActiveTemp=new ArrayList<Integer>();
		ArrayList<ArrayList<Integer>> relationship=network.neighbor;
		ArrayList<Integer> newActive=diffusionState.newActive;
		boolean[] state=diffusionState.state;
		
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
				//diffusionState.edge_record.replace(new Key(cseed,cseede), false, true);
				double probability=network.get_prob(cseed,cseede);
				//System.out.println(probability);
				if(network.isSuccess(probability))
				{  
					if(!state[cseede])
					{
						state[cseede]=true;
						newActiveTemp.add(cseede);
						diffusionState.aNum++;
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
		
	}*/


	
	

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		

	   

	}

}

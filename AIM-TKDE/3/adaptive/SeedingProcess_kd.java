package adaptive;


import java.util.ArrayList;

import adaptive.Policy.Command_k;

public class SeedingProcess_kd{
	
	static double regret_ratio=Double.MIN_VALUE;
	static boolean sign_regret_ratio=false;
	static int round;
	
	public static void MultiGo(Network network, Command_k command, int simutimes, int k, int d, ArrayList<Double> record)
	{
		//ArrayList<Double> c_result=new ArrayList<Double>();
		//System.out.println("MultiGo");
		double result=0;
		for(int i=0; i<network.vertexNum; i++)
		{
			record.add(0.0);
		}
		for(int i=0; i<simutimes; i++)
		{
			//c_result.clear();
			if(i % 5 ==0)
			{
				System.out.print(i+" ");
			}
			
			result=result+Go(network, command, k, d, record, round);
		}
		System.out.println();
		for(int i=0; i<network.vertexNum; i++)
		{
			record.set(i, record.get(i)/simutimes);
		}
		//System.out.println(result/simutimes);
	}
	
	public static double Go(Network network, Command_k command, int k, int d, ArrayList<Double> record, int round)
	{
		//System.out.println("Go");
		DiffusionState diffusionState=new DiffusionState(network, network.vertexNum, k);
		double influence=0;
		for(int i=0; i<round; i++)
		{
			if(d==0 && diffusionState.budget_left>0)
			{
				
				
				ArrayList<Integer> seed_set=new ArrayList<Integer>();
				seed_set=command.compute_seed_set(network, diffusionState,k);
				if(sign_regret_ratio && d>0)
				{
					//------------
					double c_ratio=diffusionState.estimate_regret_ratio(network, seed_set, command);
					if(c_ratio>regret_ratio)
					{
						regret_ratio=c_ratio;
					}
					//
				}
				diffusionState.seed(seed_set);
			}
			if( d>0 && i % d==0 && diffusionState.budget_left>0)
			{
				ArrayList<Integer> seed_set=new ArrayList<Integer>();
				seed_set=command.compute_seed_set(network, diffusionState,1);
				//Tools.printlistln(seed_set);
				//System.out.println("seeding done ");
				if(sign_regret_ratio && d>0)
				{
					//------------
					double c_ratio=diffusionState.estimate_regret_ratio(network, seed_set, command);
					if(c_ratio>regret_ratio)
					{
						regret_ratio=c_ratio;
					}
					//
				}
				diffusionState.seed(seed_set);
				
			}
			influence=diffusionState.diffuse(network, 1);
			//System.out.println(i+" "+diffusionState.aNum);
			record.set(i, record.get(i)+diffusionState.aNum);
		}

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

package adaptive2;


import java.util.ArrayList;

import adaptive2.Policy.Command;
import java.util.concurrent.Callable;

public class SeedingProcess_kd implements Callable<Double> {

	static double regret_ratio=Double.MIN_VALUE;
    private Network network;
    private Command command;
    private int k;
    private int d;
    private ArrayList<Double> record;

    public SeedingProcess_kd(Network network, Command command, int k, int d, ArrayList<Double> record){
        this.network = network;
        this.command = command;
        this.k = k;
        this.d = d;
        this.record = record;
    }

	public static void MultiGo(Network network, Command command, int simutimes, int k, int d, ArrayList<Double> record)
	{
		//ArrayList<Double> c_result=new ArrayList<Double>();
		System.out.println("MultiGo");
		double result=0;
		for(int i=0; i<k*d; i++)
		{
			record.add(0.0);
		}
		for(int i=0; i<simutimes; i++)
		{
			//c_result.clear();
			if(i %1 ==0)
			{
				System.out.println("Simulation number "+i);
			}

			result=result+Go(network, command, k, d, record);
		}
		for(int i=0; i<k*d; i++)
		{
			record.set(i, record.get(i)/simutimes);
		}
		System.out.println(result/simutimes);
	}

	public static double Go(Network network, Command command, int k, int d,ArrayList<Double> record)
	{
		//System.out.println("Go");
		DiffusionState diffusionState=new DiffusionState(network, network.vertexNum, -1);
		double influence=0;
		for(int i=0; i<k; i++)
		{
			//System.out.println("Round "+i+"-"+round);
			ArrayList<Integer> seed_set=new ArrayList<Integer>();
			seed_set=command.compute_seed_set(network, diffusionState);
			//Tools.printlistln(seed_set);
			//System.out.println("seed set size "+seed_set.size());
			//------------
			double c_ratio=diffusionState.estimate_regret_ratio(network, seed_set, command);		// slow
			if(c_ratio>regret_ratio)
			{
				regret_ratio=c_ratio;
			}
			//
			diffusionState.seed(seed_set);
			for(int j=0;j<d;j++)
			{
				diffusionState.diffuse(network, 1);
				record.set(i*d+j, record.get(i*d+j)+diffusionState.aNum);
			}

		}
		//System.out.println();
		influence=diffusionState.exp_influence_complete(network, 1000);
		return influence;

	}

    public Double call(){
        long threadId = Thread.currentThread().getId();
        System.out.printf("Current running thread #%03d\n", threadId);
        return Go(this.network, this.command, this.k, this.d, this.record);
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

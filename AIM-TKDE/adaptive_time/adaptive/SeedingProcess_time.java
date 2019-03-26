package adaptive;


import java.util.ArrayList;

import adaptive.Policy.Command;

public class SeedingProcess_time{

	public static int round=-1;
    public static long startTime = 0, endTime = 0;

	public static void MultiGo(
			Network network,
			Command command,
			int simutimes,
			int budget,
			ArrayList<Double> record,
			ArrayList<Integer> record_budget, String type, int d)
	{
		//ArrayList<Double> c_result=new ArrayList<Double>();
		System.out.println("MultiGo");
		if(round==-1)
		{
			throw new ArithmeticException("round = -1");
		}
		for(int i=0; i<round; i++)
		{
			record.add(0.0);
			record_budget.add(0);
		}
		double result=0;
		for(int i=0; i<simutimes; i++)
		{
			//c_result.clear();
			System.out.println("Simulation number "+i);
			switch(type)
			{
				case "dynamic":
					result=result+Go_dynamic(network, command, round, budget,record,record_budget);
					break;
				case "static":
					result=result+Go_static(network, command, round, budget,record,record_budget);
					break;
				case "uniform":
					result=result+Go_uniform_d(network, command, round, d, budget,record, record_budget);
					break;
				default:
					System.out.print("Invalid model");
			}

			result=result+Go_dynamic(network, command, round, budget,record,record_budget);
		}

		for(int i=0; i<round; i++)
		{
			record.set(i, record.get(i)/simutimes);
			record_budget.set(i, record_budget.get(i)/simutimes);
		}
		System.out.println(result/simutimes);
	}

	public static double Go_dynamic(Network network, Command command, int round, int budget, ArrayList<Double> record, ArrayList<Integer> record_budget)
	{
		//System.out.println("Go");
		DiffusionState diffusionState=new DiffusionState(network, round, budget);
		double influence=0;
		for(int i=0; i<round; i++)
		{
			//System.out.println("Round "+i+"-"+round);
			ArrayList<Integer> seed_set=new ArrayList<Integer>();
            startTime = System.currentTimeMillis();

			seed_set=command.compute_seed_set(network, diffusionState,0);

            endTime = System.currentTimeMillis();
            Tools.printElapsedTime(startTime, endTime, "compute_seed_set in dynamic");
			//Tools.printlistln(seed_set);
			//System.out.println("seed set size "+seed_set.size());
			diffusionState.seed(seed_set);
			influence=diffusionState.diffuse(network, 1);
			record.set(i, record.get(i)+diffusionState.aNum);
			record_budget.set(i, record_budget.get(i)+seed_set.size());
		}
		//System.out.println();
		return influence;

	}

	public static double Go_uniform_d(Network network, Command command, int round, int d, int budget, ArrayList<Double> record,ArrayList<Integer> record_budget)
	{
		//System.out.println("Go");
		DiffusionState diffusionState=new DiffusionState(network, round, budget);
		double influence=0;
        boolean measureTime = true;
		for(int i=0; i<round; i++)
		{
			//System.out.println("Round "+i+"-"+round);
			ArrayList<Integer> seed_set=new ArrayList<Integer>();
			if(i % d==0 && diffusionState.budget_left>0)
			{
                if(measureTime){
                    startTime = System.currentTimeMillis();
                }
				seed_set=command.compute_seed_set(network, diffusionState, Math.min(budget/(round/d), diffusionState.budget_left));
                if(measureTime){
                    endTime = System.currentTimeMillis();
                    Tools.printElapsedTime(startTime, endTime, "compute_seed_set in uniform");
                    measureTime = false;
                }
				diffusionState.seed(seed_set);
			}

			//ArrayList<Integer> seed_set=new ArrayList<Integer>();
			//seed_set=command.compute_seed_set(network, diffusionState);
			//Tools.printlistln(seed_set);
			//System.out.println("seed set size "+seed_set.size());

			influence=diffusionState.diffuse(network, 1);
			record.set(i, record.get(i)+diffusionState.aNum);
			record_budget.set(i, record_budget.get(i)+seed_set.size());
		}
		//System.out.println();
		return influence;

	}

	public static double Go_static(Network network, Command command, int round, int budget, ArrayList<Double> record,ArrayList<Integer> record_budget)
	{
		//System.out.println("Go");
		DiffusionState diffusionState=new DiffusionState(network, round, budget);
		double influence=0;
		for(int i=0; i<round; i++)
		{

			ArrayList<Integer> seed_set=new ArrayList<Integer>();
			if(i==0)
			{
                startTime = System.currentTimeMillis();
				seed_set=command.compute_seed_set(network, diffusionState, budget);
                endTime = System.currentTimeMillis();
                Tools.printElapsedTime(startTime, endTime, "compute_seed_set in static");
				diffusionState.seed(seed_set);
			}


			influence=diffusionState.diffuse(network, 1);
			record.set(i, record.get(i)+diffusionState.aNum);
			record_budget.set(i, record_budget.get(i)+seed_set.size());
		}
		//System.out.println();
		return influence;

	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub




	}

}

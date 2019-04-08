package adaptive;


import java.util.ArrayList;

import adaptive.Policy.Command;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.locks.ReentrantLock;

public class SeedingProcess_time{

	public static int round=-1;
    private static ExecutorService pool = null;
    private static ArrayList<Future<Double>> results = new ArrayList<>();

    public static void createThreadPool(){
        pool = Executors.newFixedThreadPool(36);
    }

    public static void shutdownThreadPool(){
        pool.shutdown();
    }

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
        createThreadPool();
		for(int i=0; i<round; i++)
		{
			record.add(0.0);
			record_budget.add(0);
		}
		double result=0;
        ArrayList<ArrayList<Double>> records = new ArrayList<>();
        ArrayList<ArrayList<Integer>> records_budget = new ArrayList<>();
		for(int i=0; i<simutimes; i++)
		{
			//c_result.clear();
			System.out.println("Simulation number "+i);
            records.add(new ArrayList<Double>());
            records_budget.add(new ArrayList<Integer>());
            ArrayList<Double> _record = records.get(i);
            ArrayList<Integer> _record_budget = records_budget.get(i);
			switch(type)
			{
				case "dynamic":
                    results.add(pool.submit(()->{
                        return Go_dynamic(network, command, round, budget, _record, _record_budget);
                    }));
					// result=result+Go_dynamic(network, command, round, budget,record,record_budget);
					break;
				case "static":
                    results.add(pool.submit(()->{
                        return Go_static(network, command, round, budget, _record, _record_budget);
                    }));
					// result=result+Go_static(network, command, round, budget,record,record_budget);
					break;
				case "uniform":
                    results.add(pool.submit(()->{
                        return Go_uniform_d(network, command, round, d, budget, _record, _record_budget);
                    }));
					// result=result+Go_uniform_d(network, command, round, d, budget,record, record_budget);
					break;
				default:
					System.out.print("Invalid model");
			}
		}

        for(Future<Double> future: results){
            try{
                result += future.get();
            }
            catch(InterruptedException e){
                System.out.println("Interrupted");
            }
            catch(ExecutionException e){
                System.out.println("Fail to get the result");
                e.printStackTrace();
            }
        }

        for(ArrayList<Double> arr: records){
    		for(int i=0; i<round; i++){
    			record.set(i, record.get(i)+arr.get(i));
    		}
        }
        for(ArrayList<Integer> arr: records_budget){
    		for(int i=0; i<round; i++){
    			record_budget.set(i, record_budget.get(i)+arr.get(i));
    		}
        }
        for(int i = 0;i < round; i++){
            record.set(i, record.get(i)/simutimes);
            record_budget.set(i, record_budget.get(i)/simutimes);
        }
		System.out.println(result/simutimes);
        shutdownThreadPool();
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
			seed_set=command.compute_seed_set(network, diffusionState,0);
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
		for(int i=0; i<round; i++)
		{
			//System.out.println("Round "+i+"-"+round);
			ArrayList<Integer> seed_set=new ArrayList<Integer>();
			if(i % d==0 && diffusionState.budget_left>0)
			{
				seed_set=command.compute_seed_set(network, diffusionState, Math.min(budget/(round/d), diffusionState.budget_left));
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
				seed_set=command.compute_seed_set(network, diffusionState, budget);
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

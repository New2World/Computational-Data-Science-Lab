package adaptive;


import java.util.ArrayList;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import adaptive.Policy.Command_k;

public class SeedingProcess_kd implements Callable<Doubld> {

	static double regret_ratio=Double.MIN_VALUE;
	static boolean sign_regret_ratio=false;
	static int round;
	static int ratio_times;

    private static Network network;
    private static Command_k command;
    private static int k;
    private static int d;
    private ArrayList<Double> _record;
    private static ExecutorService pool = null;
    private static ArrayList<Future<Double>> results = new ArrayList<Future<Double>>();

	// public static void MultiGo(Network network, Command_k command, int simutimes, int k, int d, ArrayList<Double> record)
	// {
	// 	//ArrayList<Double> c_result=new ArrayList<Double>();
	// 	//System.out.println("MultiGo");
	// 	double result=0;
	// 	for(int i=0; i<network.vertexNum; i++)
	// 	{
	// 		record.add(0.0);
	// 	}
	// 	for(int i=0; i<simutimes; i++)
	// 	{
	// 		//c_result.clear();
	// 		if(i % 5 ==0)
	// 		{
	// 			System.out.print(i+" ");
	// 		}
    //
	// 		result=result+Go(network, command, k, d, record, round);
	// 	}
	// 	System.out.println();
	// 	for(int i=0; i<network.vertexNum; i++)
	// 	{
	// 		record.set(i, record.get(i)/simutimes);
	// 	}
	// 	//System.out.println(result/simutimes);
	// }

    public SeedingProcess_kd(ArrayList<Double> _record){
        this._record = _record;
    }

    public static void createThreadPool(){
        pool = Executors.newFixedThreadPool(36);
    }

    public static void shutdownThreadPool(){
        pool.shutdown();
    }

    public static void MultiGo(Network network, Command_k command, int simutimes, int k, int d, ArrayList<Double> record){
        double result = 0.;
        long startTime = System.currentTimeMillis();
        long elapsedTime = startTime;
        for(int i = 0;i < round;i++){
            record.add(0.);
        }
        SeedingProcess_kd.network = network;
        SeedingProcess_kd.command = command;
        SeedingProcess_kd.k = k;
        SeedingProcess_kd.d = d;
        ArrayList<ArrayList<Double>> recordList = new ArrayList<ArrayList<Double>>();

        // TODO
        for(int i = 0;i < simutimes;i++){
            ArrayList<Double> tempRecord = new ArrayList<Double>();
            for(int j = 0;j < round;j++){
                tempRecord.add(0.);
            }
            recordList.add(tempRecord);
            Callable<Double> process = new SeedingProcess_kd(recordList.get(i));
            results.add(pool.submit(process));
        }

        for(Future<Double> future: results){
            try{
                result += future.get();
                elapsedTime = System.currentTimeMillis() - startTime;
                // System.out.printf("[*] Time: %02d:%02d:%02d.%03d\n", elapsedTime/3600000, elapsedTime/60000%60, elapsedTime/1000%60, elapsedTime%1000);
            }
            catch(InterruptedException e){
                System.out.println("Interrupted");
            }
            catch(ExecutionException e){
                System.out.println("Fail to get the result");
                e.printStackTrace();
            }
        }

        for(int i = 0;i < round;i++){
            for(ArrayList<Double> arr: recordList){
                record.set(i, record.get(i)+arr.get(i));
            }
            record.set(i, record.get(i)/simutimes);
        }
        elapsedTime = System.currentTimeMillis() - startTime;
        System.out.printf("> elapsed time: %02d:%02d:%02d.%03d\n", elapsedTime/3600000, elapsedTime/60000%60, elapsedTime/1000%60, elapsedTime%1000);
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
				diffusionState.seed(seed_set);
			}
			if( d>0 && i % d==0 && diffusionState.budget_left>0)
			{
				ArrayList<Integer> seed_set=new ArrayList<Integer>();
				seed_set=command.compute_seed_set(network, diffusionState,1);
				//Tools.printlistln(seed_set);
				//System.out.println("seeding done ");
				if(sign_regret_ratio && d>0 && i>0)
				{
					//------------
					double c_ratio=diffusionState.estimate_regret_ratio(network, seed_set, command, ratio_times);
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

    public Double call(){
        return Go(network, command, k, d, _record, round);
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

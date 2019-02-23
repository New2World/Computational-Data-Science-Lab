package adaptive2;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;


public class Start {

	public static void main(String[] args){
		// TODO Auto-generated method stub
		String wiki="../wiki_test_data/wiki.txt";

		int d=3;
		int k=3;
		int simutimes=100;
        int n_threads = 10;
        long startTime = 0;
        long runningTime = 0;
        double result = 0.;
		Network network=new Network(wiki, "IC" , 8300);		// WC
		network.set_ic_prob(0.1);		// 0.01

		ArrayList<Double> record=new ArrayList<Double>();
        ExecutorService pool = Executors.newFixedThreadPool(n_threads);
        ArrayList<Future<Double>> results = new ArrayList<Future<Double>>();

        startTime = System.currentTimeMillis();
        for(int i = 0;i < simutimes;i++){
            Callable<Double> process = new SeedingProcess_kd(network, new Policy.Greedy_policy_kd(), k, d, record);
            results.add(pool.submit(process));
        }
        for(Future<Double> future: results){
            try{
                result += future.get();
                runningTime = System.currentTimeMillis() - startTime;
                System.out.printf("[*] Time: %02d:%02d:%02d.%03d\n", runningTime/3600000, runningTime/60000%60, runningTime/1000%60, runningTime%1000);
            }
            catch(InterruptedException e){
                System.out.println("Interrupted");
            }
            catch(ExecutionException e){
                System.out.println("Fail to get the result");
            }
        }

		//SeedingProcess_new.MultiGo(network, new Policy.Greedy_policy(), simutimes, round, budget);

		//SeedingProcess_new.MultiGo(network, new Policy.Random_policy(), simutimes, round, budget);

		// SeedingProcess_kd.MultiGo(network, new Policy.Greedy_policy_kd(), simutimes, k, d, record);

		//network.sort_by_degree();
		//SeedingProcess_kd.MultiGo(network, new Policy.Degree_policy_kd(), simutimes, k, d, record);

		//SeedingProcess_kd.MultiGo(network, new Policy.Random_policy_kd(), simutimes, k, d, record);
        runningTime = System.currentTimeMillis() - startTime;
        System.out.println("---");
		System.out.println(SeedingProcess_kd.regret_ratio);
		Tools.printdoublelistln(record);
        System.out.printf("Elapsed Time: %02d:%02d:%02d.%03d\n", runningTime/3600000, runningTime/60000%60, runningTime/1000%60, runningTime%1000);

        pool.shutdown();
	}

}

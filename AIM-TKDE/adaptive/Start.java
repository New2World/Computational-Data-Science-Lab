package adaptive;

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

		int round=5;
		int budget=4;
		int simutimes=4;
        int n_threads = 4;
        double result = 0.;
		Network network=new Network(wiki, "IC" , 8300);

        ExecutorService pool = Executors.newFixedThreadPool(n_threads);
        ArrayList<Future<Double>> results = new ArrayList<Future<Double>>();

        for(int i = 0;i < simutimes;i++){
            Callable<Double> process = new SeedingProcess_new(network, new Policy.Greedy_policy(), round, budget);
            results.add(pool.submit(process));
            System.out.printf("submit task #%3d\n", i+1);
        }
        System.out.println("------------");
        for(Future<Double> future: results){
            try{
                result += future.get();
                System.out.println("# Get");
            }
            catch(InterruptedException e){
                e.printStackTrace();
            }
            catch(ExecutionException e){
                e.printStackTrace();
            }
        }
        System.out.println("------------");
        System.out.println(result/simutimes);

        pool.shutdown();

		// SeedingProcess_new.MultiGo(network, new Policy.Greedy_policy(), simutimes, round, budget);
		//SeedingProcess_new.MultiGo(network, new Policy.Random_policy(), simutimes, round, budget);
	}

}

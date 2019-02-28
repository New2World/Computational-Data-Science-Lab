package adaptive;

import java.util.ArrayList;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class Start{

	public static void run(String name, int simutimes, int k, int d, int rrestsize, int vnum)
	{
		String wiki="../../wiki_test_data/"+name+".txt";

		//int d=1;
		//int k=5;
		//int simutimes=1000;
		Network network=new Network(wiki, "WC" , vnum);
		network.set_ic_prob(0.1);



		SeedingProcess_kd.round=(k+1)*(d+1)+10;
		Policy.rrsets_size=rrestsize;
		//SeedingProcess_kd.sign_regret_ratio=true;

		System.out.println("k d "+k+" "+d);
		System.out.println("simutimes "+simutimes);
		System.out.println("rrsets_size "+Policy.rrsets_size);

		ArrayList<Double> record;
        	SeedingProcess_kd.createThreadPool();

		SeedingProcess_kd.sign_regret_ratio=true;
		System.out.println("---");
		System.out.println("greedy");
		record=new ArrayList<Double>();
		SeedingProcess_kd.MultiGo(network, new Policy.Greedy_policy_kd(), simutimes, k, d, record);
		System.out.println(SeedingProcess_kd.regret_ratio);
		Tools.printdoublelistln(record, (k+1)*(d+1)+10);

		SeedingProcess_kd.sign_regret_ratio=false;
		System.out.println("---");
		System.out.println("degree");
		record=new ArrayList<Double>();
		network.sort_by_degree();
		SeedingProcess_kd.MultiGo(network, new Policy.Degree_policy_kd(), simutimes, k, d, record);
		Tools.printdoublelistln(record, (k+1)*(d+1)+10);

		System.out.println("---");
		System.out.println("random");
		record=new ArrayList<Double>();
		SeedingProcess_kd.MultiGo(network, new Policy.Random_policy_kd(), simutimes, k, d, record);
		Tools.printdoublelistln(record, (k+1)*(d+1)+10);

        	SeedingProcess_kd.shutdownThreadPool();
		System.out.println("------------------------------------------------------");
	}

	public static void main(String[] args){
		// TODO Auto-generated method stub
		int d=0;
		int k=5;
		int simutimes=100;
		int rrset_size=100000;


		//SeedingProcess_kd.round=(k+1)*d;


		String name="wiki";
		int vnum=8300;        // 35000
        long startTime = System.currentTimeMillis();
		for(int i=0;i<5;i++)
		{
			d=2*i;
			Start.run(name, simutimes, k, d, rrset_size, vnum);
		}
        long runningTime = System.currentTimeMillis() - startTime;
        System.out.println("---");
        System.out.printf("Elapsed Time: %02d:%02d:%02d.%03d\n", runningTime/3600000, runningTime/60000%60, runningTime/1000%60, runningTime%1000);
	}

}

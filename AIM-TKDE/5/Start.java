package adaptive;

import java.util.ArrayList;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class Start{

	public static void run(String name, int simutimes, int k, int d, int rrestsize, int vnum, Network network, int ratio_times)
	{

		SeedingProcess_kd.round=(k+1)*(d+1)+10;
		SeedingProcess_kd.ratio_times=ratio_times;
		Policy.rrsets_size=rrestsize;
		//SeedingProcess_kd.sign_regret_ratio=true;

		System.out.println("k d "+k+" "+d);
		System.out.println("simutimes "+simutimes);
		System.out.println("rrsets_size "+Policy.rrsets_size);
		System.out.println("ratio_times "+ratio_times);

		ArrayList<Double> record;
        SeedingProcess_kd.createThreadPool();

		SeedingProcess_kd.sign_regret_ratio=false;
		System.out.println("greedy");
		record=new ArrayList<Double>();
		SeedingProcess_kd.MultiGo(network, new Policy.Greedy_policy_kd(), simutimes, k, d, record);
		System.out.println(SeedingProcess_kd.regret_ratio);
		Tools.printdoublelistln(record, (k+1)*(d+1)+10);

		SeedingProcess_kd.sign_regret_ratio=false;
		System.out.println("degree");
		record=new ArrayList<Double>();
		network.sort_by_degree();
		SeedingProcess_kd.MultiGo(network, new Policy.Degree_policy_kd(), simutimes, k, d, record);
		Tools.printdoublelistln(record, (k+1)*(d+1)+10);


		System.out.println("radnom");
		record=new ArrayList<Double>();
		SeedingProcess_kd.MultiGo(network, new Policy.Random_policy_kd(), simutimes, k, d, record);
		Tools.printdoublelistln(record, (k+1)*(d+1)+10);

        SeedingProcess_kd.shutdownThreadPool();
		System.out.println("------------------------------------------------------");
	}

	public static void main(String[] args){
		// TODO Auto-generated method stub
		// double d=0.5;
		// int k=5;
		// int simutimes=500;
		int rrset_size=100000;
		// int ratio_times=100;

		//SeedingProcess_kd.round=(k+1)*d;


		//String name="wiki";
		//int vnum=8300;
		//String type="WC";

		String name=args[0];          // higgs
        String type=args[1];          // VIC
		int vnum=Integer.parseInt(args[2]);             // 10000
        int simutimes = Integer.parseInt(args[3]);      // 500
        int ratio_times = Integer.parseInt(args[4]);    // 100
        int d = Integer.parseInt(args[5]);
        int k = Integer.parseInt(args[6]);

		//String name="hepph";
		//int vnum=35000;
		//String type="WC";

		//String name="hepth";
		//int vnum=27770;
		//String type="WC";

		//String name="youtube";
		//int vnum=1157900;


		String path="data/"+name+".txt";
		Network network=new Network(path, type , vnum);
		network.set_ic_prob(0.1);

        long startTime = System.currentTimeMillis();

		Start.run(name, simutimes, k, 0, rrset_size, vnum, network, ratio_times);
		for(int i=0;i<5;i++)
		{
			d=Math.pow(2, i);
			Start.run(name, simutimes, k, (int)d, rrset_size, vnum, network, ratio_times);
		}

        long elapsedTime = System.currentTimeMillis() - startTime;
        System.out.printf("Elapsed Time: %02d:%02d:%02d.%03d\n", elapsedTime/3600000, elapsedTime/60000%60, elapsedTime/1000%60, elapsedTime%1000);
	}

}

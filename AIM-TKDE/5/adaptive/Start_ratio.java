package adaptive;

import java.util.ArrayList;

public class Start_ratio{

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

		SeedingProcess_kd.sign_regret_ratio=true;
		System.out.println("greedy");
		record=new ArrayList<Double>();
		SeedingProcess_kd.MultiGo(network, new Policy.Greedy_policy_kd(), simutimes, k, d, record);
		System.out.println(SeedingProcess_kd.regret_ratio);
		Tools.printdoublelistln(record, (k+1)*(d+1)+10);

        SeedingProcess_kd.shutdownThreadPool();
		System.out.println("------------------------------------------------------");
	}

	public static void main(String[] args){
		// TODO Auto-generated method stub
		int rrset_size=100000;
		int ratio_times=100;

		//SeedingProcess_kd.round=(k+1)*d;


        String name=args[0];          // higgs
        String type=args[1];          // VIC
		int vnum=Integer.parseInt(args[2]);             // 10000
        int simutimes = Integer.parseInt(args[3]);      // 500
        int d = Integer.parseInt(args[4]);
        int k = Integer.parseInt(args[5]);

		// String name="higgs";
		// int vnum=10000;
		// String type="VIC";

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

		Start_ratio.run(name, simutimes, k, 1, rrset_size, vnum, network, ratio_times);




	}

}

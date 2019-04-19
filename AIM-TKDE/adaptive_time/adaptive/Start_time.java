package adaptive;

import java.util.ArrayList;

public class Start_time{

    public static void run(String name, int simutimes, int k,  int vnum, Network network)
    {
        //SeedingProcess_kd.sign_regret_ratio=true;
        long startTime, endTime;
        long launchTime, terminateTime;

        System.out.println("k d "+k+" "+SeedingProcess_time.round);
        System.out.println("simutimes "+simutimes);
        System.out.println("simu_rest_times "+Policy.simurest_times);
        System.out.println("rrsets_size "+Policy.rrsets_size);

        ArrayList<Double> record;
        ArrayList<Integer> record_budget;

        SeedingProcess_time.createThreadPool();

        launchTime = System.currentTimeMillis();
        System.out.println("-------------------------------------------");
        System.out.println("-------------------------------------------");
        System.out.println("dynamic");
        record=new ArrayList<Double>();
        record_budget=new ArrayList<Integer>();
        startTime = System.currentTimeMillis();
        SeedingProcess_time.MultiGo(network, new Policy.Greedy_policy_dynamic(), simutimes, k, record,record_budget,"dynamic",-1);
        endTime = System.currentTimeMillis();
        Tools.printElapsedTime(startTime, endTime, "dynamic");
        Tools.printdoublelistln(record, SeedingProcess_time.round);
        Tools.printintlistln(record_budget, SeedingProcess_time.round);
        System.out.println("-------------------------------------------");
        System.out.println("-------------------------------------------");

        System.out.println("static");
        record=new ArrayList<Double>();
        record_budget=new ArrayList<Integer>();
        startTime = System.currentTimeMillis();
        SeedingProcess_time.MultiGo(network, new Policy.Greedy_policy_kd(), simutimes, k, record,record_budget,"static",-1);
        endTime = System.currentTimeMillis();
        Tools.printElapsedTime(startTime, endTime, "static");
        Tools.printdoublelistln(record, SeedingProcess_time.round);
        Tools.printintlistln(record_budget, SeedingProcess_time.round);
        System.out.println("-------------------------------------------");
        System.out.println("-------------------------------------------");


        System.out.println("uniform 1");
        record=new ArrayList<Double>();
        record_budget=new ArrayList<Integer>();
        startTime = System.currentTimeMillis();
        SeedingProcess_time.MultiGo(network, new Policy.Greedy_policy_kd(), simutimes, k, record,record_budget,"uniform",1);
        endTime = System.currentTimeMillis();
        Tools.printElapsedTime(startTime, endTime, "uniform 1");
        Tools.printdoublelistln(record, SeedingProcess_time.round);
        Tools.printintlistln(record_budget, SeedingProcess_time.round);
        System.out.println("-------------------------------------------");
        System.out.println("-------------------------------------------");


        System.out.println("uniform 2");
        record=new ArrayList<Double>();
        record_budget=new ArrayList<Integer>();
        startTime = System.currentTimeMillis();
        SeedingProcess_time.MultiGo(network, new Policy.Greedy_policy_kd(), simutimes, k, record,record_budget,"uniform",2);
        endTime = System.currentTimeMillis();
        Tools.printElapsedTime(startTime, endTime, "uniform 2");
        Tools.printdoublelistln(record, SeedingProcess_time.round);
        Tools.printintlistln(record_budget, SeedingProcess_time.round);
        System.out.println("-------------------------------------------");
        System.out.println("-------------------------------------------");

        System.out.println("uniform 5");
        record=new ArrayList<Double>();
        record_budget=new ArrayList<Integer>();
        startTime = System.currentTimeMillis();
        SeedingProcess_time.MultiGo(network, new Policy.Greedy_policy_kd(), simutimes, k, record,record_budget,"uniform",5);
        endTime = System.currentTimeMillis();
        Tools.printElapsedTime(startTime, endTime, "uniform 5");
        Tools.printdoublelistln(record, SeedingProcess_time.round);
        Tools.printintlistln(record_budget, SeedingProcess_time.round);
        System.out.println("-------------------------------------------");
        System.out.println("-------------------------------------------");

        terminateTime = System.currentTimeMillis();
        Tools.printElapsedTime(launchTime, terminateTime);

        SeedingProcess_time.shutdownThreadPool();
    }

    public static void main(String[] args){
        // TODO Auto-generated method stub
        //double d=0.5;


        SeedingProcess_time.round=5;
        Policy.simurest_times=100;         // 100
        Policy.rrsets_size=100000;
        //int ratio_times=100;

        //SeedingProcess_kd.round=(k+1)*d;

        String name=args[0];
        String type=args[1];
        int vnum=Integer.parseInt(args[2]);
        int simutimes = Integer.parseInt(args[3]);                  // 100
        SeedingProcess_time.round = Integer.parseInt(args[4]);      // 5
        Policy.simurest_times = Integer.parseInt(args[5]);          // 100
        int k = Integer.parseInt(args[6]);                          // 20

        // String name="wiki";
        // int vnum=8300;
        // String type="WC";

        //String name="higgs";
        //int vnum=10000;
        //String type="VIC";

        //String name="hepph";
        //int vnum=35000;
        //String type="WC";

        //String name="hepth";
        //int vnum=27770;
        //String type="WC";

        //String name="dblp";
        //int vnum=430000;
        //String type="WC";

        //String name="youtube";
        //int vnum=1157900;


        String path="data/"+name+".txt";
        Network network=new Network(path, type , vnum);
        network.set_ic_prob(0.1);

        Start_time.run(name, simutimes, k, vnum, network);



    }

}

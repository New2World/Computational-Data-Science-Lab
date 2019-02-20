package adaptive;



public class Start {

	public static void main(String[] args){
		// TODO Auto-generated method stub
		String wiki="data/wiki.txt";

		int round=5;
		int budget=4;
		int simutimes=100;		
		Network network=new Network(wiki, "IC" , 8300);

		SeedingProcess_new.MultiGo(network, new Policy.Greedy_policy(), simutimes, round, budget);
		//SeedingProcess_new.MultiGo(network, new Policy.Random_policy(), simutimes, round, budget);
	}

}

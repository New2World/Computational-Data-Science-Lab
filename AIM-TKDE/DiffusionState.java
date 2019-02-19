package adaptive;


import java.util.ArrayList;
import java.util.Random;

public class DiffusionState {

	//public ArrayList<ArrayList<Boolean>> edge_record;
	public boolean[] state;
	public ArrayList<Integer> newActive;
	public int round_left, budget_left;
	public boolean round_limit=true;;
	//public HashMap<Key, Boolean> edge_record=new HashMap<Key, Boolean>();
	public int aNum;
	
	public DiffusionState(Network network,int round_left, int budget_left)
	{
		//edge_record=new ArrayList<ArrayList<Boolean>>();
		state=new boolean[network.vertexNum];
		newActive =new ArrayList<Integer>();
		this.round_left=round_left;
		this.budget_left=budget_left;
		if(round_left==network.vertexNum)
		{
			round_limit=false;
		}
		aNum=0;
		
		for(int i=0;i<network.vertexNum;i++)
		{
			state[i]=false;
		}
		/*
		for(int i=0;i<network.neighbor.size();i++)
		{
			int node=i;
			for(int j=0;j<network.neighbor.get(i).size();i++)
			{
				int neighbor=network.neighbor.get(i).get(j);
				Key key=new Key(node,neighbor);
				edge_record.put(key, false);
			}
		}*/
	}


	public DiffusionState(DiffusionState diffusionState) {
		state=diffusionState.state.clone();
		newActive =new ArrayList<Integer>(diffusionState.newActive);
		this.round_left=diffusionState.round_left;
		this.budget_left=diffusionState.budget_left;
		round_limit=diffusionState.round_limit;
		aNum=diffusionState.aNum;
		// TODO Auto-generated constructor stub
	}

	public double diffuse(Network network, int round)
	{
		for(int i=0;i<round;i++)
		{
			diffuse_one_round(network);
		}
		return aNum;
	}
	
	
	private void diffuse_one_round(Network network)
	{
		ArrayList<Integer> newActiveTemp=new ArrayList<Integer>();
		ArrayList<ArrayList<Integer>> relationship=network.neighbor;
		Random rand=new Random();
		//double result=0;

		for(int i=0;i<newActive.size();i++)
		{
			
			int cseed=newActive.get(i);
			ArrayList<Integer> cseed_neighbor=relationship.get(cseed);
			
			for(int j=0;j<cseed_neighbor.size();j++)
			{
				int cseede=cseed_neighbor.get(j);
				double probability=network.get_prob(cseed,cseede);
	
				if(rand.nextFloat() < probability)
				{  
					if(!state[cseede])
					{						
						state[cseede]= true;
						aNum++;
						//result++;
						newActiveTemp.add(cseede);
					}
				}

			}
			
			
		}
		newActive.clear();
		for(int i=0;i<newActiveTemp.size();i++)   
		{
			newActive.add(newActiveTemp.get(i));
		}
		
		round_left--;

	}
	
	public void seed(ArrayList<Integer> seed_set) {
		
		if(seed_set.size()>budget_left)
		{
			throw new ArithmeticException("diffusionstate.seed over budgetd"); 
		}
		for(int i=0;i<seed_set.size();i++)
		{
			if(!state[seed_set.get(i)])
			{
				state[seed_set.get(i)]=true;
				newActive.add(seed_set.get(i));
				aNum++;
			}
			else
			{
				
				throw new ArithmeticException("diffusionstate.seed: seeding an active node "+seed_set.get(i)); 
			}
				
		}
		budget_left=budget_left-seed_set.size();
		
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
	}


	

}

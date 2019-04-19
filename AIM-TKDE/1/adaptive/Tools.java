package adaptive;

import java.util.ArrayList;

public class Tools {
	public static void printlistln(ArrayList<Integer> list)
	{
		for(int i=0;i<list.size();i++)
		{
			System.out.print(list.get(i)+" ");
		}
		System.out.println();
	}
	
}
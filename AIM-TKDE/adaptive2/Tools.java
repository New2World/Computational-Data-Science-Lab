package adaptive2;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;

public class Tools {
	static NumberFormat formatter = new DecimalFormat("#0.00");
	//System.out.println(formatter.format(4.0));
	public static void printlistln(ArrayList<Integer> list)
	{
		for(int i=0;i<list.size();i++)
		{
			System.out.print(list.get(i)+" ");
		}
		System.out.println();
	}

	public static void printdoublelistln(ArrayList<Double> list)
	{
		for(int i=0;i<list.size();i++)
		{
			System.out.print(formatter.format(list.get(i))+" ");
		}
		System.out.println();
	}

}
package adaptive;

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
			if(i>0 && (list.get(i)-list.get(i-1))<0.01)
			{
				System.out.print(formatter.format(list.get(i))+" ");
				System.out.println();
				return;
			}
			else
			{
				System.out.print(formatter.format(list.get(i))+" ");
			}
			
		}
		System.out.println();
	}
	
	public static void printdoublelistln(ArrayList<Double> list, int size)
	{
		if(size>list.size())
		{
			throw new ArithmeticException("printdoublelistln size>list.size()"); 
			
		}
		for(int i=0;i<size;i++)
		{
			
			if(i<list.size())
			{
				System.out.print(formatter.format(list.get(i))+" ");
			}
			
			
		}
		System.out.println();
	}
	
}
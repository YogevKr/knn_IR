import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class Utils {
    public static ArrayList<String> fileToLineList(File file){

        ArrayList<String> list = new ArrayList<String>();
        try {
            Scanner input = new Scanner(file);

            while (input.hasNextLine()) {
                list.add(input.nextLine());
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        return list;
    }

    public static ArrayList<String> fileToLineList(String filePath){
        return fileToLineList(new File(filePath));
    }


    public static ArrayList<String[]> ReadCsvFile(String filePath) {

        String line = "";
        String cvsSplitBy = ",";

        ArrayList<String[]> out = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {

            while ((line = br.readLine()) != null) {

                out.add(line.split(cvsSplitBy));
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        return out;
    }
}

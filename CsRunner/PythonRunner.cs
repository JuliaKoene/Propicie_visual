using System;
using System.Diagnostics;

public class PythonRunner 
{
    static void StartPythonScript(string scriptName)
    {
        ProcessStartInfo psi = new ProcessStartInfo();
        psi.FileName = "python"; 
        psi.Arguments = scriptName;  
        psi.UseShellExecute = false;
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;
        psi.CreateNoWindow = true;

        Process process = new Process();
        process.StartInfo = psi;
        process.Start();

        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();

        Console.WriteLine($"Output from {scriptName}:\n" + output);
        if (!string.IsNullOrEmpty(error))
        {
            Console.Error.WriteLine($"Error in {scriptName}:\n" + error);
        }

        process.WaitForExit();
    }

    static void  Main()
    {
        StartPythonScript("../Sit-and-Reach/sit_and_reach_holistic_2.py");

        StartPythonScript("../Back-Scratch/back_scratch.py");
    }
}

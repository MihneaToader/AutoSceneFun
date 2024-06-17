using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.IO;
using System.Threading.Tasks;
using System;


public class AudioRecorder : MonoBehaviour
{
    private AudioClip recordedClip;
    private bool isRecording = false;
    private string fileName = "recorded_audio.wav";
    private string filePath;

    private void Start()
    {
        Debug.Log(string.Join(", ", Microphone.devices));
        StartRecording();
    }


    private void StartRecording()
    {
        isRecording = true;
        StartCoroutine(RecordAudio());
    }

    private void StopRecording()
    { 
        isRecording = false;
        StopCoroutine(RecordAudio());
        SaveRecording();
    }

    private IEnumerator RecordAudio()
    {
        var device = Microphone.devices[0];
        recordedClip = Microphone.Start(device, true, 10, AudioSettings.outputSampleRate);
        while (Microphone.GetPosition(device) <= 0)
        {
            yield return null;
        }

        while (isRecording)
        {
            yield return null;
        }

        Microphone.End(device);
        StopRecording();
    }

    private void SaveRecording()
    {
        filePath = $"{Application.persistentDataPath}/{System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")}_{fileName}";
        WaveFile.Save(filePath, recordedClip);
    }
   
    private void OnApplicationPause()
    {
        if (isRecording)
        {
            SaveRecording();
        }
    }
     
}

public static class WaveFile
{
    public static void Save(string filePath, AudioClip clip)
    {
        Debug.Log(string.Join(", ", Microphone.devices));
        var samples = new float[clip.samples * clip.channels];
        clip.GetData(samples, 0);
        var byteCount = samples.Length * 2;
        var headerSize = 44;
        var dataSize = byteCount;
        var riffSize = dataSize + headerSize - 8;

        using (var file = File.OpenWrite(filePath))
        {
            // Write the WAV header
            file.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"), 0, 4);
            file.Write(BitConverter.GetBytes(riffSize), 0, 4);
            file.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"), 0, 4);
            file.Write(System.Text.Encoding.ASCII.GetBytes("fmt "), 0, 4);
            file.Write(BitConverter.GetBytes(16), 0, 4);
            file.Write(BitConverter.GetBytes((short)1), 0, 2);
            file.Write(BitConverter.GetBytes((short)clip.channels), 0, 2);
            file.Write(BitConverter.GetBytes(clip.frequency), 0, 4);
            file.Write(BitConverter.GetBytes((int)(clip.frequency * clip.channels * 2)), 0, 4);
            file.Write(BitConverter.GetBytes((short)(clip.channels * 2)), 0, 2);
            file.Write(BitConverter.GetBytes((short)16), 0, 2);
            file.Write(System.Text.Encoding.ASCII.GetBytes("data"), 0, 4);
            file.Write(BitConverter.GetBytes(dataSize), 0, 4);

            // Write the audio data
            for (int i = 0; i < samples.Length; i++)
            {
                file.Write(BitConverter.GetBytes((short)(samples[i] * short.MaxValue)), 0, 2);
            }
        }
    }
}
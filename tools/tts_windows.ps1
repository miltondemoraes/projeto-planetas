# Sintetiza frases em WAV com as vozes do Windows (sem internet).
# Uso: powershell -ExecutionPolicy Bypass -File tts_windows.ps1 -Out arquivo.wav -Text "frase" [-Voice Daniel]
# Tenta a API moderna (WinRT / OneCore: voz masculina "Daniel" pt-BR); se falhar, usa SAPI (System.Speech).
param([string]$Out, [string]$Text, [string]$Voice = "Daniel")
$ErrorActionPreference = "Stop"
try {
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    $asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
    [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime] | Out-Null
    $synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
    $v = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.DisplayName -like "*$Voice*" -and $_.Language -like "pt*" }
    if (-not $v) { $v = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.Language -like "pt*" } }
    if ($v) { $synth.Voice = @($v)[0] }
    $op = $synth.SynthesizeTextToStreamAsync($Text)
    $task = $asTaskGeneric.MakeGenericMethod([Windows.Media.SpeechSynthesis.SpeechSynthesisStream]).Invoke($null, @($op))
    $task.Wait(-1) | Out-Null
    $stream = [System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($task.Result)
    $fs = [System.IO.File]::Create($Out)
    $stream.CopyTo($fs)
    $fs.Close()
    Write-Output ("winrt:" + @($v)[0].DisplayName)
}
catch {
    Add-Type -AssemblyName System.Speech
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $pt = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like "pt*" }
    if ($pt) { $s.SelectVoice(@($pt)[0].VoiceInfo.Name) }
    $s.SetOutputToWaveFile($Out)
    $s.Speak($Text)
    $s.Dispose()
    Write-Output "sapi"
}

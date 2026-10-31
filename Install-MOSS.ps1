[CmdletBinding()]
param([string]$Python, [string]$ComfyRoot, [switch]$NonInteractive)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$installLog = Join-Path $env:TEMP ('MOSS-install-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
Add-Type -AssemblyName System.Windows.Forms
$source = $PSScriptRoot

function Show-Message([string]$text) {
    Write-Host $text
    if (-not $NonInteractive) { [void][System.Windows.Forms.MessageBox]::Show($text, 'MOSS 安装', 'OK', 'Information') }
}

try {
    if (-not $Python -or -not $ComfyRoot) {
        $instances = @()
        $probe = Join-Path $source 'tools\inspect_comfy_process.py'
        $processes = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '\bmain\.py\b' })
        foreach ($process in $processes) {
            # A Windows venv launcher owns a base-Python child with the same command.
            if ($processes.ProcessId -contains $process.ParentProcessId) { continue }
            $start = New-Object System.Diagnostics.ProcessStartInfo
            $start.FileName = $process.ExecutablePath
            $start.Arguments = '-s "' + $probe + '" ' + $process.ProcessId
            $start.UseShellExecute = $false
            $start.CreateNoWindow = $true
            $start.RedirectStandardOutput = $true
            $start.RedirectStandardError = $true
            $child = New-Object System.Diagnostics.Process
            $child.StartInfo = $start
            [void]$child.Start()
            $stdout = $child.StandardOutput.ReadToEndAsync()
            $stderr = $child.StandardError.ReadToEndAsync()
            $child.WaitForExit()
            $inventory = $stdout.Result
            if ($child.ExitCode -eq 0 -and $inventory) {
                $instance = $inventory | ConvertFrom-Json
                if (-not ($instances | Where-Object { $_.root -eq $instance.root })) { $instances += $instance }
            } else {
                Add-Content -LiteralPath $installLog -Encoding UTF8 -Value ($process.ExecutablePath + ': ' + $stderr.Result)
            }
            $child.Dispose()
        }
        if ($instances.Count -eq 1) {
            $Python = $instances[0].python; $ComfyRoot = $instances[0].root
        } elseif ($instances.Count -gt 1) {
            $form = New-Object System.Windows.Forms.Form
            $form.Text = '选择要安装 MOSS 的 ComfyUI'; $form.Width = 900; $form.Height = 260; $form.StartPosition = 'CenterScreen'
            $list = New-Object System.Windows.Forms.ListBox
            $list.Dock = 'Fill'
            foreach ($item in $instances) { [void]$list.Items.Add(('端口 {0} — {1}' -f $item.port, $item.root)) }
            $list.SelectedIndex = -1
            $button = New-Object System.Windows.Forms.Button
            $button.Text = '使用所选 ComfyUI'; $button.Dock = 'Bottom'; $button.Height = 40
            $button.DialogResult = 'OK'; $form.AcceptButton = $button
            $button.Enabled = $false
            $list.Add_SelectedIndexChanged({ $button.Enabled = $list.SelectedIndex -ge 0 })
            $form.Controls.Add($list); $form.Controls.Add($button)
            if ($form.ShowDialog() -ne 'OK') { exit 0 }
            $Python = $instances[$list.SelectedIndex].python; $ComfyRoot = $instances[$list.SelectedIndex].root
            $form.Dispose()
        } else {
            if ($NonInteractive) { throw '请指定当前 ComfyUI 的 Python 和根目录。' }
            $picker = New-Object System.Windows.Forms.FolderBrowserDialog
            $picker.Description = '选择 ComfyUI_windows_portable、Desktop 的实例目录，或包含 main.py 的源码目录。启动 ComfyUI 后也可重新运行安装器自动识别。'
            if ($picker.ShowDialog() -ne 'OK') { exit 0 }
            $chosen = $picker.SelectedPath; $picker.Dispose()
            if (Test-Path (Join-Path $chosen 'python_embeded\python.exe')) {
                $Python = Join-Path $chosen 'python_embeded\python.exe'; $ComfyRoot = Join-Path $chosen 'ComfyUI'
            } elseif (Test-Path (Join-Path (Split-Path $chosen) 'python_embeded\python.exe')) {
                $Python = Join-Path (Split-Path $chosen) 'python_embeded\python.exe'; $ComfyRoot = $chosen
            } elseif (Test-Path (Join-Path $chosen '.venv\Scripts\python.exe')) {
                $Python = Join-Path $chosen '.venv\Scripts\python.exe'; $ComfyRoot = $chosen
            } elseif (Test-Path (Join-Path $chosen 'venv\Scripts\python.exe')) {
                $Python = Join-Path $chosen 'venv\Scripts\python.exe'; $ComfyRoot = $chosen
            } else {
                $pythonPicker = New-Object System.Windows.Forms.OpenFileDialog
                $pythonPicker.Title = '选择这个 ComfyUI 环境的 python.exe（支持 Conda 和外置虚拟环境）'
                $pythonPicker.Filter = 'Python interpreter|python.exe'
                if ($pythonPicker.ShowDialog() -ne 'OK') { exit 0 }
                $Python = $pythonPicker.FileName; $ComfyRoot = $chosen
                $pythonPicker.Dispose()
            }
        }
    }
    if (-not (Test-Path -LiteralPath $Python) -or -not (Test-Path -LiteralPath $ComfyRoot)) { throw '所选 ComfyUI 目录不存在。' }
    $destination = Join-Path $ComfyRoot 'custom_nodes\kohaku-moss-lowvram-optimized'
    $message = "将安装或更新 MOSS 到：`n$destination`n`n使用该 ComfyUI 自己的 Python：`n$Python`n`n不安装第二个 Manager，不更换 Torch，不下载模型。请先结束当前生成任务。"
    if (-not $NonInteractive -and [System.Windows.Forms.MessageBox]::Show($message, '确认安装 MOSS', 'OKCancel', 'Information') -ne 'OK') { exit 0 }
    Write-Host $message
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    if ([System.IO.Path]::GetFullPath($source) -ne [System.IO.Path]::GetFullPath($destination)) {
        Get-ChildItem -LiteralPath $source -Force | Where-Object { $_.Name -notin @('.git', '__pycache__', 'node.zip', 'models.local.json') } | Copy-Item -Destination $destination -Recurse -Force
    }
    Write-Host '正在准备节点运行组件，保留当前 ComfyUI 与 Torch……'
    & $Python -s -u (Join-Path $destination 'install.py') --log-file $installLog
    if ($LASTEXITCODE -ne 0) { throw 'MOSS 运行环境尚未准备完成。详细原因已显示在窗口中，不能把此状态视为安装成功。' }
    Show-Message 'MOSS 安装完成。现在关闭并重新启动 ComfyUI，刷新网页，再拖入同文件夹中的 example_workflows\00_moss_lowvram.json。准备向导会自动打开。选择已有本地模型即可运行；缺少模型时只勾选需要的文件下载。'
} catch {
    Write-Host ($_.Exception.Message + "`n安装日志：$installLog") -ForegroundColor Red
    if (-not $NonInteractive) { [void][System.Windows.Forms.MessageBox]::Show(($_.Exception.Message + "`n安装日志：$installLog"), 'MOSS 尚未安装完成', 'OK', 'Error') }
    exit 1
}

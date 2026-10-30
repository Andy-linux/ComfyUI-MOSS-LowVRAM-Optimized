[CmdletBinding()]
param([string]$Python, [string]$ComfyRoot, [switch]$NonInteractive)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Windows.Forms
$source = $PSScriptRoot

function Show-Message([string]$text) {
    Write-Host $text
    if (-not $NonInteractive) { [void][System.Windows.Forms.MessageBox]::Show($text, 'MOSS 安装', 'OK', 'Information') }
}

try {
    if (-not $Python -or -not $ComfyRoot) {
        $instances = @()
        $probe = @'
import json, pathlib, psutil, sys
p = psutil.Process(int(sys.argv[1]))
a = p.cmdline()
cwd = pathlib.Path(p.cwd())
main = next(pathlib.Path(x) for x in a[1:] if pathlib.Path(x).name == 'main.py')
main = main if main.is_absolute() else cwd / main
root = pathlib.Path(a[a.index('--base-directory') + 1]) if '--base-directory' in a else main.parent
root = root if root.is_absolute() else cwd / root
port = int(a[a.index('--port') + 1]) if '--port' in a else 8188
print(json.dumps({'python': sys.executable, 'root': str(root.resolve()), 'port': port}))
'@
        foreach ($process in (Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '\bmain\.py\b' })) {
            $inventory = & $process.ExecutablePath -s -c $probe $process.ProcessId 2>$null
            if ($LASTEXITCODE -eq 0 -and $inventory) {
                $instance = $inventory | ConvertFrom-Json
                if (-not ($instances | Where-Object { $_.root -eq $instance.root })) { $instances += $instance }
            }
        }
        if ($instances.Count -eq 1) {
            $Python = $instances[0].python; $ComfyRoot = $instances[0].root
        } elseif ($instances.Count -gt 1) {
            $form = New-Object System.Windows.Forms.Form
            $form.Text = '选择要安装 MOSS 的 ComfyUI'; $form.Width = 760; $form.Height = 240
            $list = New-Object System.Windows.Forms.ListBox
            $list.Dock = 'Fill'
            foreach ($item in $instances) { [void]$list.Items.Add(('端口 {0} — {1}' -f $item.port, $item.root)) }
            $list.SelectedIndex = 0
            $button = New-Object System.Windows.Forms.Button
            $button.Text = '使用所选 ComfyUI'; $button.Dock = 'Bottom'; $button.Height = 40
            $button.DialogResult = 'OK'; $form.AcceptButton = $button
            $form.Controls.Add($list); $form.Controls.Add($button)
            if ($form.ShowDialog() -ne 'OK') { exit 0 }
            $Python = $instances[$list.SelectedIndex].python; $ComfyRoot = $instances[$list.SelectedIndex].root
            $form.Dispose()
        } else {
            if ($NonInteractive) { throw '请指定当前 ComfyUI 的 Python 和根目录。' }
            $picker = New-Object System.Windows.Forms.FolderBrowserDialog
            $picker.Description = '请先启动 ComfyUI；也可在这里选择 ComfyUI_windows_portable 或包含 main.py 的源码安装文件夹。'
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
            } else { throw '没有找到这个 ComfyUI 使用的 Python。请先启动 ComfyUI，再双击安装器；会从运行中的程序自动识别，不使用系统 Python。' }
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
    Write-Host '正在安装节点所需的 Python 依赖……'
    & $Python -s -m pip install --disable-pip-version-check -r (Join-Path $destination 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Python 依赖安装未完成。请保留此窗口的错误信息，确认网络后重新运行安装器；不会重装 ComfyUI。' }
    & $Python -s (Join-Path $destination 'install.py')
    if ($LASTEXITCODE -ne 0) { throw 'MOSS 运行环境尚未准备完成。详细原因已显示在窗口中，不能把此状态视为安装成功。' }
    Show-Message 'MOSS 安装完成。现在关闭并重新启动 ComfyUI，刷新网页，再拖入同文件夹中的 example_workflows\00_moss_lowvram.json。准备向导会自动打开，按提示确认模型下载后即可运行。'
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    if (-not $NonInteractive) { [void][System.Windows.Forms.MessageBox]::Show($_.Exception.Message, 'MOSS 尚未安装完成', 'OK', 'Error') }
    exit 1
}

# 设置工作目录
$env:WS = "E:\Xuke\kaggle\Benetech"

function Hold-on-PyTask {
    $Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 1) -RepetitionDuration (New-TimeSpan -Days 365)
    $Action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-ExecutionPolicy Bypass -Command "& {ip-update}"'
    # 注册计划任务，指定任务名称、触发器、动作、设置和主体
    Register-ScheduledTask -TaskName "Hold on PyTask" -Trigger $Trigger -Action $Action 
}
function Kill-PyTask {
    # 注销任务
    Unregister-ScheduledTask -TaskName "Hold on PyTask" -Confirm:$false
}

function PyTask {
    download-file

    # 切换到工作目录
    cd $env:WS

    # 激活虚拟环境
    & "$env:WS\venv\scripts\activate.ps1"

    # 执行 Python 代码
    # python log_stdout.py 3>&1 2>&1 >> "$env:WS\loggings\example_stdout.log"
    python train_matcha.py 3>&1 2>&1 >> "$env:WS\loggings\example_stdout.log"
}

function download-file {
    # get the latest json file
    $base_url = "http://20.163.99.216:8080"
    $url = "$base_url/download/ip/"
    $env:WS = "E:\Xuke\kaggle\Benetech"
    $output = "$env:WS\loggings\example_data.bak.json"
    Invoke-WebRequest -Uri $url -OutFile $output
}

# Hold-On-PyTask
PyTask
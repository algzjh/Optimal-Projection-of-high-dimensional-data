/**
 * Created by Administrator on 2017/12/19.
 */

//读取数据
//读取数据
function onFileSelected(){
	var dataName = document.getElementById("selectData").value;  //获取用户选择的文件名
	var metricName = document.getElementById("selectMetric").value;
	var threshold = document.getElementById("Threshold").value;
    var evaluation = document.getElementById("selectEvaluation").value;
   
	InitSystem();       //清除所有面板
	loadData(dataName, metricName, threshold, evaluation); //导入数据

}

//数据初始化
function InitSystem(){
	//清空数组
	originData = [];   //原始数据
	distMatrix = [];   //原始数据
    dataDim = 0;
    dataSize = 0;
    dataClass = 0;
}

//加载数据
function loadData(dataName, metricName, threshold, evaluation){
    /*
    //var csv = d3.dsv(",", "text/csv;charset=gb2312"); // csv 文件的编码格式，根据需要进行更改
    var csv = d3.dsv(",", "text/csv;charset=gb2312"); 
    // csv 文件的编码格式，根据需要进行更改
 

	csv(dataName,function(error,data) {
        if (error) throw error;
        console.log(data);
        console.log("read data successfully! ")
        dataTransform(data);
        //var selsect = document.getElementById("selectData");
        //var name = selsect.options[selsect.selectedIndex].text;
	});
	*/
	/*
	console.log(dataName);
	
	d3.text(dataName, function(text){
		console.log(d3.csv.parseRows(text));
	})
    */
	
    /*
	d3.text(dataName, function(text){
		var data = d3.csv.parseRows(text).map(function(row){
			return row.map(function(value){
				return + value;
			});
		});
		console.log(data);
        console.log("read data successfully! ");
        dataTransform(data, metricName, threshold, dataName);
	})
    */
    dataTransform(metricName, threshold, dataName, evaluation);
	
}

function countUnique(iterable) {
  return new Set(iterable).size;
}

//数据格式的处理
function dataTransform(metricName, threshold, dataName, evaluation){
    /*
    var vec = [];
    var originLabel = [];
    for (var i in data) {
    	var first = true;
        for (var j in data[i]) {
            //vec.push(parseFloat(data[i][j]));
            if (first){
            	first = false;
            	originLabel.push(data[i][j]);
            }else{
            	vec.push(data[i][j]);
            }
        }
        originData.push(vec);
        vec = [];
    }
    console.log("originLabel: ");
    console.log(originLabel);
    console.log("originData:");
    console.log(originData);
    //console.log(data);
    dataDim = originData[0].length;
    dataSize = originData.length;
    console.log("originData:");
    console.log(originData);
    console.log("dataDim:");
    console.log(dataDim);
    console.log("dataSize:");
    console.log(dataSize);

    document.getElementById("DataSize").value = dataSize;
    document.getElementById("Dimension").value = dataDim;
    document.getElementById("DataClass").value = countUnique(originLabel)
    */
        //获取Lamp中的Y YS
    //originData = normalized(originData)
    
    dataDeal(metricName, threshold, dataName, evaluation);//不归一化
}

function normalized(Data){
	for(var i = 0; i < Data[0].length; i ++){
		var max = Data[0][i];
		var min = Data[0][i];
		for(var j = 0 ; j < Data.length; j ++){
			if(Data[j][i] > max){
				max = Data[j][i];
			}
			if(Data[j][i] < min){
				min = Data[j][i];
			}
		}
		if(max-min == 0){
			for(var j = 0 ; j < Data.length; j ++) {
				Data[j][i] = 0;
			}
			continue;
		}
		var div = 1/(max-min);
		for(var j = 0 ; j < Data.length; j ++) {
			Data[j][i] = (Data[j][i] - min) * div;
		}
	}
	return Data;
}

function onPartitionAlgChange(){
    var partitionName = document.getElementById("selectPalg").value;
    if(partitionName=='MDS'){
        //document.getElementById("pLDA5Plot").innerHTML="MDS生成结果"
        document.getElementById("pLDA5Plot").innerHTML="pLDA5"
    }else{
        document.getElementById("pLDA5Plot").innerHTML="pLDA5"
    }

    
}

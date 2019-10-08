/**
 * Created by Administrator on 2017/12/19.
 */

function drawinfo2(gongarray){
    d3.select('#metricchart').select("svg").remove();
    var svg = d3.select('#metricchart')
    .append('svg')
    .attr('width','990')
    .attr('height','80')

    //var dataset = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    var dataset = gongarray
    console.log("gongarray")
    console.log(gongarray)
    var min = d3.min(dataset)
    var max = d3.max(dataset)

    var linear = d3.scale.linear()
    .domain([0, max])
    .range([0, 800]);

    var rects = svg.selectAll('rect') //选择svg内所有的矩形
    .data(dataset) //绑定数组
    .enter() //指定选择集的enter部分
    .append('rect') //添加足够数量的矩形元素
    .attr('x','10') //矩形距离左上角的x坐标
    .attr('y',function(d,i){return i*12}) //矩形距离左上角的x坐标
    // .on("mouseover",function(d,i){
    // d3.select(this)
    // .attr("fill","yellow");
    // }) //光标放在矩形上时填充变色
    // .on("mouseout",function(d,i){
    // d3.select(this)
    // .transition()
    // .duration(500)
    // .attr("fill","steelblue");
    // }) //光标离开矩形上时填充变色
    // .transition() //启用动画
    // .duration(2000) //过渡持续时间
    .attr('width',function(d){return linear(d)}) //宽度按比例尺给定
    .attr('height',8)
    .attr("fill","steelblue")       //填充颜色不要写在CSS里steelblue
    


    var texts = svg.selectAll("text")
    .data(dataset)
    .enter()
    .append("text")
    .attr("class","MyText")
    .attr("x", function(d){return linear(d)+12}) //左上角
    .attr("y",function(d,i){return i*12+8}) //基线（文字底部）
    .text(function(d, i){
        var algName = "tmp"
        if(i==0){
            algName = "PCA "
        }else if(i==1){
            algName = "MDS "
        }else if(i==2){
            algName = "LDA "
        }else if(i==3){
            algName = "Isomap "
        }else if(i==4){
            algName = "t-SNE "
        }else if(i==5){
            algName = "pLDA "
        }
    return algName + d;
    });

    var axis = d3.axisBottom()//V4更改，比例尺的位置
    .scale(linear)
    .ticks(4);//指定比例尺刻度
     
    svg.append("g")
    .attr("class","axis")
    .attr("transform","translate(20,130)")
    .call(axis); //axis(svg)


}

function drawinfo(){
    // my add begin

    var data = [1,4,7,2,9,13],
    bar_height = 6,
    bar_padding = 3,
    svg_height = (bar_height + bar_padding)*data.length,
    svg_width = 900;

    var scale = d3.scale.linear()
    .domain([0, d3.max(data)])
    .range([0, svg_width])

    var svg = d3.select("#metricchart")
    .append("svg")
    .attr("width", svg_width)
    .attr("height", svg_height)
    .attr("margin-top", "5px")

    var bar = svg.selectAll("g")
    .data(data)
    .enter()
    .append("g")
    .attr("transform",function(d,i){return "translate(0,"+ i*(bar_height+bar_padding) +")";})
     
    bar.append("rect")
    .attr({
        "width":function(d){return scale(d);},
        "height":bar_height
    })
    .style("fill","steelblue")
     
    bar.append("text")
    .text(function(d){return d;})
    .attr({
        "x":function(d){return scale(d);},
        "y":bar_height/2,
        "text-anchor":"end"
    })
    
    // my add end
}

function dataDeal(metricName, threshold, dataName, evaluation) {
    console.log("dataDeal threshold")
    console.log(threshold)


    var formData = new FormData();
    formData.append("id", "requestPlot");
    formData.append("csrfmiddlewaretoken", token);
    //console.log(formData.get("id"));
    //console.log(formData.get("csrfmiddlewaretoken"));

    //formData.append("originData", JSON.stringify(originData));
    //formData.append("originLabel", JSON.stringify(originLabel));
    formData.append("originMetric", metricName);
    formData.append("threshold", threshold);
    formData.append("dataName", dataName);
    formData.append("evaluation", evaluation)
    console.log("dataName: ");
    console.log(formData.get("dataName"));



    $.ajaxSetup({
        data: {csrfmiddlewaretoken: '{{ csrf_token }}'},
    });

    var Y_pca;
    var Y_mds;
    var Y_lda;
    var Y_isomap;
    var label;
    $.ajax(
            {
                url:"http://127.0.0.1:8000/graapp/",
                type:"POST",
                data:formData,
                processData: false,
                contentType: false,
                success:function(data){
                    jsonData = $.parseJSON(data);
                    // Ys = jsonData.Ys;
                    Y_pca = jsonData.Y_pca;
                    Y_mds = jsonData.Y_mds;
                    Y_lda = jsonData.Y_lda;
                    Y_isomap = jsonData.Y_isomap;
                    Y_tsne = jsonData.Y_tsne;
                    //label = jsonData.label;
                    label = jsonData.label;
                    Y_pLDA = jsonData.Y_pLDA;

                    console.log("label");
                    console.log(label);

                    console.log("Y_pca:");
                    console.log(Y_pca);
                    console.log("Y_mds:");
                    console.log(Y_mds);
                    console.log("Y_tsne");
                    console.log(Y_tsne);
                    console.log("Y_pLDA");
                    console.log(Y_pLDA);

                    dataDim = jsonData.dataDim
                    dataSize = jsonData.dataSize
                    dataClass = jsonData.dataClass
                    console.log('dataDim: ')
                    console.log(dataDim)
                    console.log('dataSize: ')
                    console.log(dataSize)
                    console.log('dataClass: ')
                    console.log(dataClass)

                    document.getElementById("DataSize").value = dataSize;
                    document.getElementById("Dimension").value = dataDim;
                    document.getElementById("DataClass").value = dataClass;


                    drawReduce(Y_pca,label,"pcaPlot");//用函数画,green
                    drawReduce(Y_mds,label,"mdsPlot");//用函数画,green
                    drawReduce(Y_lda,label,"ldaPlot");//用函数画,green
                    drawReduce(Y_isomap,label,"isomapPlot");//用函数画,green
                    drawReduce(Y_tsne, label, "tsnePlot");

                    view_num = jsonData.view_num;
                    console.log("view_num");
                    console.log(view_num);
                    for (var i = 1; i <= view_num; i++) {
                        if(i == 1){
                            drawReduce(jsonData.Y_pLDA1,jsonData.Y_pLDA1_Label, "pLDA1Plot");
                        }else if(i == 2){
                            drawReduce(jsonData.Y_pLDA2,jsonData.Y_pLDA2_Label, "pLDA2Plot");
                        }else if(i == 3){
                            drawReduce(jsonData.Y_pLDA3,jsonData.Y_pLDA3_Label, "pLDA3Plot");
                        }else if(i == 4){
                            drawReduce(jsonData.Y_pLDA4,jsonData.Y_pLDA4_Label, "pLDA4Plot");
                        }else if(i == 5){
                            drawReduce(jsonData.Y_pLDA5,jsonData.Y_pLDA5_Label, "pLDA5Plot");
                        }
                    }

                    var gongarray = new Array(6)
                    gongarray[0] = jsonData.gong_pca
                    gongarray[1] = jsonData.gong_mds
                    gongarray[2] = jsonData.gong_lda
                    gongarray[3] = jsonData.gong_isomap
                    gongarray[4] = jsonData.gong_tsne
                    gongarray[5] = jsonData.gong_pLDA
                    drawinfo2(gongarray);

                },
                error:function(data){
                    alert("异常！");
                }
            });
}
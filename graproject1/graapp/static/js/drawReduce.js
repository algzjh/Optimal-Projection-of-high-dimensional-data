/**
 * Created by Administrator on 2017/12/20.
 */


function drawReduce(data,label,id) {

    for(var i = 0; i < data.length; i ++){
        data[i][0] = parseFloat(data[i][0]);
        data[i][1] = parseFloat(data[i][1]);
    }

    console.log(label)
    console.log(data)

    //获得数据
    //id = "mdsPlot";
    d3.select('#' + id).select("svg").remove();
    var Width = 205;
    //var Width = 1400;
    var Height = 170;
    var padding1 = {left: 30, right: 0, top: 0, bottom: 20};

    console.log('#' + id);
    var dvplotSvg = d3.select('#' + id)
        .append("svg")
        .attr("id", id + "Svg")
        .attr("width", Width)
        .attr("height", Height)
        ;

    var minx = d3.min(data, function(d,i){
        return d[0];
    });
    console.log("minx: ")
    console.log(minx)
    var maxx = d3.max(data, function(d,i){
        return d[0];
    });
    console.log("maxx: ")
    console.log(maxx)
    var miny = d3.min(data, function(d,i){
        return d[1];
    });
    console.log("miny: ")
    console.log(miny)
    var maxy = d3.max(data, function(d,i){
        return d[1];
    });
    console.log("maxy: ")
    console.log(maxy)


    var xAxisWidth = Width - padding1.left - padding1.right;
    var xScale = d3.scale.linear()
        .domain([minx, maxx])
        .range([0, xAxisWidth]);


    var yAxisWidth = Height - padding1.top - padding1.bottom;
    var yScale = d3.scale.linear()
        .domain([miny, maxy])
        .range([0, yAxisWidth]);


    console.log(label);

    var circles = dvplotSvg.selectAll("circle")
        .data(data)
        .enter()
        .append("circle")
        .attr("fill", function (d, i) {
            /*
            ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd']
            ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
            */
            if(label[i]==0) return '#a6cee3';
            if(label[i]==1) return '#1f78b4';
            if(label[i]==2) return '#b2df8a';
            if(label[i]==3) return '#33a02c';
            if(label[i]==4) return '#fb9a99';
            if(label[i]==5) return '#e31a1c';
            if(label[i]==6) return '#fdbf6f';
            if(label[i]==7) return '#ff7f00';
            if(label[i]==8) return '#cab2d6';
            if(label[i]==9) return '#6a3d9a';

        })
        .attr("opacity", 0.9)
        .attr("id", function(d, i){
            return "gdCircle" + i;
        })
        .attr("cx", function (d, i) {
            return padding1.left + xScale(d[0]);
        })
        .attr("cy", function(d, i){
            return Height - padding1.bottom - yScale(d[1]);
        })
        .attr("r", 2);
/*
    
    var xAxis = d3.svg.axis()
        .scale(xScale)
        .orient("bottom");
    yScale.range([yAxisWidth,0])
    var yAxis = d3.svg.axis()
        .scale(yScale)
        .orient("left");
    dvplotSvg.append("g")
        .attr("class","axis")
        .attr("transform","translate(" + padding1.left + "," +
            (Height - padding1.bottom) + ")")
        .call(xAxis);
    dvplotSvg.append("g")
        .attr("class","axis")
        .attr("transform","translate(" + padding1.left + "," +
            (padding1.bottom) + ")")
        .call(yAxis);
*/
    

}

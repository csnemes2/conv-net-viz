
var hidediv = null;

function showdiv(div_id)
{
    if(hidediv) { hidediv.style.display = 'none'; }
    console.log(div_id)
    hidediv = document.getElementById(div_id);
    hidediv.style.display = 'block';
}

function disp_channel(path, i,  max, pxsize=200) {
        channel_name=i
        if (channel_name=="None"){channel_name="All"}

        document.write("<table><tr><th>channel="+channel_name+"</th><th>activation</th><th>original</th><th>vizualization</th></tr>");
        document.write("<tr><th>receptive area</th><td></td>");
        document.write("<td><img src=\""+path+max+"/" + i +"o"+ ".jpg\" /></td>");
        document.write("<td><img src=\""+path+max+"/" + i +"v"+ ".jpg\" /></td>");
        document.write("</tr><tr><th>full</h>")
        document.write("<td><img src=\""+path+max+"/" + i +"a"+ ".jpg\" /></td>");
        document.write("<td><img src=\""+path+max+"/" + i +"oh"+ ".jpg\" /></td>");
        document.write("<td><img src=\""+path+max+"/" + i +"vh"+ ".jpg\" /></td>");
        document.write("</tr></table>");
        document.write("<hr/>");
}

function disp_tensor(path,num, max,pxsize=200) {
    var i=0;

    document.write("<div id=\""+make_id(path)+"\" style=\"display: none\">")


    document.write("<h1>"+path+"</h1>");

    for (i=0;i<num;i++)
    {
        disp_channel(path,i, max, pxsize);
    }
    disp_channel(path,"None", max, pxsize);

    document.write("</div>")
}

function make_id(path)
{
    return path.replace(/\//g,'_').replace(/\./g,'point')
}

function print_divs(mode, num, mode, pixel)
{
    fLen = sub_layers.length;
    for (i = 0; i < fLen; i++) {
        disp_tensor(sub_layers[i],num, mode,pixel)
    }
}

function print_buttons(sub_layers)
{
    fLen = sub_layers.length;
    for (i = 0; i < fLen; i++) {
        document.write("<button onclick=\"showdiv('"+make_id(sub_layers[i])+"'); \">"+sub_layers[i]+"</button>");
    }
}
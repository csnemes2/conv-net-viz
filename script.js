
function myFunction(path,mode,num,ending,pxsize=28) {
    var i=0;
    if (ending=="patch") {document.write("</br>PATCH mode");}
    else{document.write("</br>FULL mode (not zooming to the best patch)");}
    document.write("</br>"+path+"</br>COLUMNS: 0.."+num+" featuremaps independently; last column: all layers</br>");
    document.write("ROWS: 0-9: Orig images best activating the featuremap. 10-19: activation pattern. 20-29: deconv visualization in input pixel space</br>");
    for (i=0;i<num;i++)
    {
        document.write("<img src=\""+path+mode+"/" + i +ending+ ".jpg\" width="+pxsize+"px/>");
    }
    document.write("<img src=\""+path+mode+"/" + "None" +ending+ ".jpg\" width="+pxsize+"px/>");
}
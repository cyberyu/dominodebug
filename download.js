var data = source.data;
var filetext = 'Question,a1,s1,a1_v,s1_v,final_label\n';
for (var i = 0; i < data['Question'].length; i++) {
    var currRow = [data['Question'][i].toString(),
                   data['a1'][i].toString(),
                   data['s1'][i].toString(),
                   data['a1_v'][i].toString(),
                   data['s1_v'][i].toString(),
                   data['final_label'][i].toString().concat('\n')];

    var joined = currRow.join();
    filetext = filetext.concat(joined);
}

var filename = 'data_result.csv';
var blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename);
} else {
    var link = document.createElement("a");
    link = document.createElement('a')
    link.href = URL.createObjectURL(blob);
    link.download = filename
    link.target = "_blank";
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'))
}

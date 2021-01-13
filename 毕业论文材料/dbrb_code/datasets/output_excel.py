import xlwt
from datetime import datetime


def write_data_to_excel(name, data):
    """
        输入数据data必须是二维数据
    """
    # 实例化一个Workbook()对象(即excel文件)
    wbk = xlwt.Workbook()
    # 新建一个名为Sheet1的excel sheet。此处的cell_overwrite_ok =True是为了能对同一个单元格重复操作。
    sheet = wbk.add_sheet('Sheet1', cell_overwrite_ok=True)
    # 获取当前日期，得到一个datetime对象如：(2016, 8, 9, 23, 12, 23, 424000)
    today = datetime.today()
    # 将获取到的datetime对象仅取日期如：2016-8-9
    today_date = datetime.date(today)
    # 遍历result中的没个元素。
    for i in range(len(data)):
        # 对result的每个子元素作遍历，
        for j in range(len(data[i])):
            # 将每一行的每个元素按行号i,列号j,写入到excel中。
            sheet.write(i, j, data[i][j])
    # 以传递的name+当前日期作为excel名称保存。
    wbk.save(name + '_' + str(today_date) + '.xls')

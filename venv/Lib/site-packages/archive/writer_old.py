
from xlrd import open_workbook
from xlwt import easyxf, Formula
from xlutils import copy as cp
from models import Result, Basin, Pipe
from constants import K_MANNING


def create_storm_tabulation(analysis, filename):
    rb = open_workbook('template/stormsewer_template.xls', formatting_info=True, on_demand=True)
    wb = cp.copy(rb)
    analysis_name = analysis.get_name()
    network_name = analysis.network.get_name()
    last_run = analysis.last_run.strftime('%a %b %d %Y %H:%M:%S')
    intensity_polynomial_name = analysis.intensity_polynomial.get_name()
    ws = wb.get_sheet(0)
    style_e_left = easyxf(
        'font: height 280, name Arial; alignment: horizontal left, vertical center; ' + \
        'borders: left medium, right medium, top medium, bottom medium;'
    )
    style_se_left = easyxf(
        'font: height 280, name Arial; alignment: horizontal left, vertical center; ' + \
        'borders: left medium, right medium, top medium, bottom thick;'
    )
    ws.write(7,  25, analysis_name,             style_e_left)
    ws.write(8,  25, network_name,              style_e_left)
    ws.write(9,  25, last_run,                  style_e_left)
    ws.write(10, 25, intensity_polynomial_name, style_se_left)

    row_start = 11
    row_increment = 3
    style = easyxf(
        'font: height 280, name Arial; alignment: horizontal center, vertical center;' + \
        'borders: left medium, right medium, top medium, bottom medium;',
        num_format_str='0.00'
    )
    style_s = easyxf(
        'font: height 280, name Arial; alignment: horizontal center, vertical center;' + \
        'borders: left medium, right medium, top medium, bottom thick;',
        num_format_str='0.00'
    )
    style_per = easyxf(
        'font: height 280, name Arial; alignment: horizontal center, vertical center;' + \
        'borders: left medium, right medium, top medium, bottom medium;',
        num_format_str='0.00%'
    )
    style_s_per = easyxf(
        'font: height 280, name Arial; alignment: horizontal center, vertical center;' + \
        'borders: left medium, right medium, top medium, bottom thick;',
        num_format_str='0.00%'
    )
    results = Result.select().where(Result.analysis == analysis).order_by(Pipe.node_1.name)
    for i, result in enumerate(results):
        basin = Basin.get(node=result.pipe.node_1)
        row_curr = row_start + row_increment * i
        ws.write_merge(row_curr,     row_curr + 1, 0,  0,  result.pipe.node_1.name,      style)
        ws.write_merge(row_curr + 2, row_curr + 2, 0,  0,  result.pipe.node_2.name,      style_s)
        ws.write_merge(row_curr,     row_curr + 2, 1,  1,  result.pipe.length,           style_s)
        ws.write_merge(row_curr,     row_curr,     2,  2,  basin.c,                      style)
        ws.write_merge(row_curr,     row_curr,     3,  3,  basin.area,                   style)
        ws.write_merge(row_curr + 1, row_curr + 2, 3,  3,  result.area,                  style_s)
        ws.write_merge(row_curr + 1, row_curr + 2, 4, 4, result.runoff_area, style_s)
        ws.write_merge(row_curr,     row_curr + 2, 5,  5,  basin.tc,                     style_s)
        ws.write_merge(row_curr,     row_curr + 2, 6,  6,  result.time,                  style_s)
        ws.write_merge(row_curr, row_curr + 2, 8, 8, result.get_y, style_s)
        ws.write_merge(row_curr,     row_curr + 2, 9,  9,  '',                           style_s)
        ws.write_merge(row_curr,     row_curr + 2, 11, 11, '',                           style_s)
        ws.write_merge(row_curr,     row_curr + 2, 12, 12, result.pipe.node_1.elevation, style_s)
        ws.write_merge(row_curr, row_curr, 14, 14, result.hgl_1, style)
        ws.write_merge(row_curr + 2, row_curr + 2, 14, 14, result.pipe.invert_1,         style_s)
        ws.write_merge(row_curr, row_curr, 15, 15, result.hgl_2, style)
        ws.write_merge(row_curr + 2, row_curr + 2, 15, 15, result.pipe.invert_2,         style_s)
        ws.write_merge(row_curr,     row_curr + 2, 17, 17, result.pipe.mannings,         style_s)
        ws.write_merge(row_curr,     row_curr + 2, 18, 18, 1,                            style_s)
        ws.write_merge(row_curr,     row_curr + 2, 19, 19, result.pipe.diameter,         style_s)
        ws.write_merge(row_curr,     row_curr + 2, 22, 22, '',                           style_s)
        ws.write_merge(row_curr,     row_curr + 2, 23, 26, '',                           style_s)

        total_c = Formula('E{0}/D{0}'.format(row_curr + 2))
        total_runoff = Formula('C{0}*D{0}'.format(row_curr + 1))
        time_section = Formula('B{0}/V{0}/60'.format(row_curr + 1))
        total_flow = Formula('I{0}*E{1}*43560/60/60/12'.format(row_curr + 1, row_curr + 2))
        hgl_clearance = Formula('M{0}-O{0}'.format(row_curr + 1))
        crown_1 = Formula('O{0}+T{1}/12'.format(row_curr + 3, row_curr + 1))
        crown_2 = Formula('P{0}+T{1}/12'.format(row_curr + 3, row_curr + 1))
        fall_hgl = Formula('O{0}-P{0}'.format(row_curr + 1))
        fall_crown = Formula('O{0}-P{0}'.format(row_curr + 2))
        fall_invert = Formula('O{0}-P{0}'.format(row_curr + 3))
        slope_hydraulic = Formula('Q{0}/B{0}'.format(row_curr + 1))
        slope_physical = Formula('Q{0}/B{1}'.format(row_curr + 2, row_curr + 1))
        velocity_actual = Formula('{0}/R{1}*((((PI()*(T{1}/12/2)^2)-((T{1}/12/2)^2*((2*ACOS(((T{1}/12/2)-(T{1}/12-(MIN(O{1}-O{2},T{1}/12))))/(T{1}/12/2)))-SIN(2*ACOS(((T{1}/12/2)-(T{1}/12-(MIN(O{1}-O{2},T{1}/12))))/(T{1}/12/2))))/2))/(2*PI()*(T{1}/12/2)-(T{1}/12/2)*(2*ACOS(((T{1}/12/2)-(T{1}/12-(MIN(O{1}-O{2},T{1}/12))))/(T{1}/12/2)))))^(2/3))*U{1}^0.5'.format(K_MANNING, row_curr + 1, row_curr + 3))
        velocity_physical = Formula('{0}/R{1}*((((PI()*(T{1}/12/2)^2)-((T{1}/12/2)^2*((2*ACOS(((T{1}/12/2)-(T{1}/12-(MIN(T{1}/12))))/(T{1}/12/2)))-SIN(2*ACOS(((T{1}/12/2)-(T12/12-(MIN(T{1}/12))))/(T{1}/12/2))))/2))/(2*PI()*(T{1}/12/2)-(T{1}/12/2)*(2*ACOS(((T{1}/12/2)-(T{1}/12-(MIN(T{1}/12))))/(T{1}/12/2)))))^(2/3))*U{2}^0.5'.format(K_MANNING, row_curr + 1, row_curr + 2))

        ws.write_merge(row_curr + 1, row_curr + 2, 2,  2,  total_c,           style_s)
        ws.write_merge(row_curr,     row_curr,     4,  4,  total_runoff,      style)
        ws.write_merge(row_curr,     row_curr + 2, 7,  7,  time_section,      style_s)
        ws.write_merge(row_curr,     row_curr + 2, 10, 10, total_flow,        style_s)
        ws.write_merge(row_curr,     row_curr + 2, 13, 13, hgl_clearance,     style_s)
        ws.write_merge(row_curr + 1, row_curr + 1, 14, 14, crown_1,           style)
        ws.write_merge(row_curr + 1, row_curr + 1, 15, 15, crown_2,           style)
        ws.write_merge(row_curr,     row_curr,     16, 16, fall_hgl,          style)
        ws.write_merge(row_curr + 1, row_curr + 1, 16, 16, fall_crown,        style)
        ws.write_merge(row_curr + 2, row_curr + 2, 16, 16, fall_invert,       style_s)
        ws.write_merge(row_curr,     row_curr,     20, 20, slope_hydraulic,   style_per)
        ws.write_merge(row_curr + 1, row_curr + 2, 20, 20, slope_physical,    style_s_per)
        ws.write_merge(row_curr,     row_curr,     21, 21, velocity_actual,   style)
        ws.write_merge(row_curr + 1, row_curr + 2, 21, 21, velocity_physical, style_s)
    ws.fit_num_pages = 1
    wb.save(filename)

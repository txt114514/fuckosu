from Traning.data_type_group import Circle, Slider, Spinner

def parse_osu(file_path):
    objects = []
    in_hitobjects = False

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line == "[HitObjects]":
                in_hitobjects = True
                continue

            if not in_hitobjects or not line:
                continue

            parts = line.split(',')

            x = int(parts[0])
            y = int(parts[1])
            t = int(parts[2])
            type_flag = int(parts[3])

            # 判断类型
            if type_flag & 1:  # Circle
                objects.append(Circle(t, t, x, y))

            elif type_flag & 2:  # Slider
                slider_info = parts[5]
                repeats = int(parts[6])

                # 解析路径
                path_raw = slider_info.split('|')[1:]
                path = [(x, y)]

                for p in path_raw:
                    px, py = map(int, p.split(':'))
                    path.append((px, py))

                # ⚠️ 这里简单假设 duration（你后面可以优化）
                t_end = t + 500

                objects.append(Slider(t, t_end, path, repeats))

            elif type_flag & 8:  # Spinner
                t_end = int(parts[5])
                objects.append(Spinner(t, t_end))

    return objects
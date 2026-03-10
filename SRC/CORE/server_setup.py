import os

from SRC.CORE._CONSTANTS import project_root_dir, PORT_MAPPING_FILE_NAME


def parse_port_mapping():
    port_mapping_file_name = PORT_MAPPING_FILE_NAME()
    file_path = f'{project_root_dir()}/{port_mapping_file_name}.txt'
    data_dict = {}
    public_ip = None
    with open(file_path, 'r') as file:
        for line in file:
            words = line.split()
            if len(words) == 0:
                break

            if 'Public' in words[0]:
                public_ip = words[2]
                continue

            internal_value = int(words[1])
            external_value = int(words[3])

            data_dict[f"{internal_value}"] = external_value

    # print(f"public_ip: {public_ip}")
    # print(f"port_mapping: {data_dict}")

    return public_ip, data_dict


def print_public_url_for_port(local_port):
    from IPython.display import display, HTML

    try:
        public_ip, data_dict = parse_port_mapping()
        external_port = data_dict[str(local_port)]
        url = f"http://{public_ip}:{external_port}"

        link_html = f'<a href="{url}" target="_blank">{url}</a>'
        display(HTML(link_html))
    except:
        print("!!NO PORT MAPPING SPECIFIED!!")
        pass
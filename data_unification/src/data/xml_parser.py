import xml.etree.ElementTree as ET

def extract_kinematic_chain(body):
    chain = [body]
    children = list(body.findall("body"))
    if not children:
        return chain
    elif len(children) == 1:
        chain += extract_kinematic_chain(children[0])
        return chain
    else:
        return chain

def parse_robot_structure(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    all_bodies = worldbody.findall(".//body")
    base_body = worldbody.find("body")
    chain = extract_kinematic_chain(base_body)
    
    joint_names = []
    for b in chain[1:]:
        j = b.find("joint")
        joint_names.append(j.get("name") if j is not None else None)
    
    joint_names = [j for j in joint_names if j is not None]
    body_names = [b.get("name") for b in chain]
    
    if 'tcp_link' not in body_names:
        body_names.append('tcp_link')
        
    return body_names[1:], joint_names


if __name__ == "__main__":
  # xml_path = 'mujoco_menagerie/franka_fr3/fr3.xml'
  # xml_path = 'mujoco_menagerie/ufactory_xarm7/xarm7_mvp.xml'
  xml_path = 'mujoco_menagerie/google_robot/robot.xml'

  # 예시 사용
  chain_names, joint_names = parse_robot_structure(xml_path)
  print("Kinematic chain:", chain_names)
  print("Joints along chain:", joint_names)

import os
import winreg

def clean_path(scope="user"):
    """
    清理 Path 中的引号和不可见字符，并打印清理前后对比。
    scope: 'user' 或 'system'
    """
    if scope == "user":
        root = winreg.HKEY_CURRENT_USER
        reg_path = r"Environment"
    elif scope == "system":
        root = winreg.HKEY_LOCAL_MACHINE
        reg_path = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
    else:
        raise ValueError("scope must be 'user' or 'system'")

    with winreg.OpenKey(root, reg_path, 0, winreg.KEY_ALL_ACCESS) as key:
        try:
            value, vtype = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            print(f"{scope} Path 未找到")
            return

        print(f"\n[{scope.upper()} Path 清理前]")
        for p in value.split(";"):
            if '"' in p or any(ord(c) < 32 for c in p):
                print("❌ 可疑:", repr(p))
            else:
                print("✅ 正常:", p)

        # 清理引号和不可见字符
        cleaned = ";".join([p.strip().strip('"') for p in value.split(";") if p.strip()])

        # 写回注册表
        winreg.SetValueEx(key, "Path", 0, vtype, cleaned)

        print(f"\n[{scope.upper()} Path 清理后]")
        for p in cleaned.split(";"):
            print("✅", p)

        print(f"\n✅ {scope} Path 已清理完毕\n{'-'*60}")

if __name__ == "__main__":
    # 清理用户 Path
    clean_path("user")

    # 清理系统 Path (需要管理员权限)
    try:
        clean_path("system")
    except PermissionError:
        print("\n⚠️ 清理系统 Path 需要管理员权限，请以管理员身份运行此脚本。")

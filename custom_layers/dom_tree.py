import json
import copy

#----- CLASS REPRESENTING DOM TREE
class DOMTree:
    def __init__(self, path):
        #-- LOAD DATA FROM FILE
        with open(path,'r') as f:
            root = json.load(f)
        
        #-- INIT VARIABLES
        self.root = root

        #-- Get auxiliary info: parent nodes, local name, get pruned version of tree
        self.addAdditionalInfo()
        self.getPrunedVersion()

    def saveTree(self, path):
        copied_root = copy.deepcopy(self.root)

        # auxiliary fields shall be removed
        fields_to_remove = ['parent', 'localName', 'prunedChildNodes', 'visibleBox']

        processing_stack = []
        processing_stack.append(copied_root)

        while len(processing_stack)!=0:
            node = processing_stack.pop()

            # remove
            for field in fields_to_remove:
                if field in node:
                    del node[field]

            # follow children
            if 'childNodes' in node:
                childNodes = node['childNodes']
                for childNode in childNodes:
                    processing_stack.append(childNode)


        print('Saving tree to '+ str(path))
        dom_content = json.dumps(copied_root, indent=4, default=str)
        with open(path, 'w+') as f:
            f.write(dom_content)


    def getLocalNodeName(self, name, count):
        return name+'['+str(count)+']'

    def getPositionedLeafNodes(self):
        ## FIND LEAVES WITH POSITION
        processing_stack = []
        res = []

        processing_stack.append(self.root)    
        while len(processing_stack)!=0:
            node = processing_stack.pop()

            # if it has children follow them
            if 'childNodes' in node:
                for childNode in node['prunedChildNodes']:
                    processing_stack.append(childNode)
            # if we have not children and element has non zero position
            else:
                if 'position' in node and ((node['position'][2]-node['position'][0])*(node['position'][3]-node['position'][1]) != 0):
                    res.append(node)
                #Get parent for multiple nested elements
                if 'position' in node['parent'] and ((node['parent']['position'][2]-node['parent']['position'][0])*(node['parent']['position'][3]-node['parent']['position'][1]) != 0):
                    res.append(node['parent'])
        return res

    def getPaths(self, node):
        paths = []

        ## if it has some identifier, i.e. unique id or class
        for elem in node if isinstance(node, (list, tuple)) else [node]:
            if 'attrs' in elem:
                attrs = elem['attrs']

                # process IDS
                if 'id' in attrs:
                    for _id in attrs['id'].split():
                        # if id is in uniques
                        if _id in self.unique_ids:
                            paths.append(ElementPath(ElementPath.RootType._id, root=_id))
                    # return paths          

                # process classes
                if 'class' in attrs:
                    for _cls in attrs['class'].split():
                        if _cls in self.unique_classes:
                            paths.append(ElementPath(ElementPath.RootType._class, root=_cls))
            
            ## if it has not parent -> it is a root
            if 'parent' not in elem:
                paths.append(ElementPath(ElementPath.RootType._dom_root))
            
            
            else:
                # get paths to parent
                parent_paths = self.getPaths(elem['parent'])
                # append local name to parent path
                for ppath in parent_paths:
                    ppath.localNames.append(elem['localName'])
                    paths.append(ppath)      

        return paths

    def getElementByOneOfPaths(self, paths):
        node = None
        for path in paths:
            node = self.getElementByPath(path)
            if node is not None:
                return node
        return node


    def getElementByPath(self, path):
        # get root node, based on the type
        if path.root_type == ElementPath.RootType._dom_root:
            node = self.root
        elif path.root_type == ElementPath.RootType._class:
            if path.root in self.unique_classes:
                node = self.unique_classes[path.root]
            else:
                return None
        elif path.root_type == ElementPath.RootType._id:
            if path.root in self.unique_ids:
                node = self.unique_ids[path.root]
            else:
                return None

        # follow localNames
        for localName in path.localNames:

            # if it has no children -> path does not exist
            if 'childNodes' not in node:
                return None
            # process children
            else:
                childNamesDict = {}
                foundMatch = False
                # for every child
                for childNode in node['childNodes']:
                    # if child local name matches with our local name -> follow it
                    if localName == childNode['localName']:
                        node = childNode
                        foundMatch = True
                        break

                # if there was no match -> path does not exist
                if not foundMatch:
                    return None

        # last node is a result
        return node

    def addAdditionalInfo(self):
        self.unique_ids = {}
        self.unique_classes = {} 

        #-- GO THROUGH DATA AND SAVE SOME DATA
        # initialize processing stack with root
        processing_stack = []
        processing_stack.append(self.root)
        classes_to_delete = set()
        ids_to_delete = set()

        while len(processing_stack)!=0:
            node = processing_stack.pop()

            ###------ IDS and Classes

            # if node has some attributes
            if 'attrs' in node:
                attrs = node['attrs']
                # process IDS
                if 'id' in attrs:
                    for _id in attrs['id'].split():
                        # if id is not in uniques
                        if _id not in self.unique_ids:
                            self.unique_ids[_id] = node
                        # if it is there, its not unique -> remove
                        else:
                            ids_to_delete.add(_id)
                            
                # process classes
                if 'class' in attrs:
                    for _cls in attrs['class'].split():
                        # if class is not in uniques
                        if _cls not in self.unique_classes:
                            self.unique_classes[_cls] = node
                        # if it is there, its not unique -> remove
                        else:
                            classes_to_delete.add(_cls)

            ###------ Children processing

            # if node has children, process and follow them
            if 'childNodes' in node:
                childNodes = node['childNodes']
                childNamesDict = {} # init child names

                for childNode in childNodes:
                    # add parent link
                    childNode['parent'] = node

                    # add local name
                    childName = childNode['name']
                    if childName in childNamesDict:
                        count = childNamesDict[childName] + 1
                    else:
                        count = 1
                    childNamesDict[childName] = count
                    localName = self.getLocalNodeName(childName, count)
                    childNode['localName'] = localName

                    # add to stack for further processing
                    processing_stack.append(childNode)

        #-- REMOVE NON-UNIQUE CLASSES AND IDS
        for _cls in classes_to_delete:
            del self.unique_classes[_cls]

        for _id in ids_to_delete:
            del self.unique_ids[_id]

        #-- PRINT SIZES OF UNIQUES
        print("# of unique ids: " + str(len(self.unique_ids)))
        print("# of unique classes: " + str(len(self.unique_classes)))

    # GETS PRUNED VERSION OF THE TREE        
    def getPrunedVersion(self):
        ### REMOVE EMPTY TEXT NODES
        ### REMOVE ELEMENTS WITH ZERO SIZE (AND WHOLE SUBTREE)
        ### REMOVE ELEMENTS WHICH ARE HIDDEN (AND WHOLE SUBTREE)

        processing_stack = []
        processing_stack.append(self.root)    

        while len(processing_stack)!=0:
            node = processing_stack.pop()

            if 'childNodes' in node:
                childNodes = node['childNodes']
                prunedChildNodes = []

                for childNode in childNodes:
                    ### REMOVE EMPTY TEXT NODES
                    if ('type' in childNode) and ('value' in childNode) and (not childNode['value'].strip()):
                        continue
                    
                   

                    ### REMOVE ELEMENTS NON TEXT ELEMENTS WHICH DO NOT HAVE COMPUTED STYLE AND USELESS TAGS
                    if ('type' not in childNode or 'trash' in childNode) and ('computed_style' not in childNode): 
                        continue

                    ### REMOVE HIDDEN ELEMENTS 
                    if ('computed_style' in childNode):
                        if (childNode['computed_style']['display'] == "none" or
                            childNode['computed_style']['visibility'] == "hidden" or
                            childNode['computed_style']['opacity'] == "0"):
                            continue

                    ### ADD TO CHILDREN AND TO PROCESSING STACK
                    prunedChildNodes.append(childNode)
                    processing_stack.append(childNode)

                ### SET NEW CHILDREN
                node['prunedChildNodes'] = prunedChildNodes

#----- CLASS REPRESENTING PATH TO ELEMENT (IN DOM TREE)
class ElementPath:
    def __init__(self, root_type, root=None, localNames=None):
        self.root_type = root_type  # whether use root of tree, or unique class, or unique id
        self.root = root            # root defines where path starts
        if localNames is None:
            localNames = []
        self.localNames = localNames
    
    def __str__(self):
        if self.root_type!=ElementPath.RootType._dom_root:
            s = '['+str(self.root_type)+':'+self.root+']'
        else:
            s = '['+str(self.root_type)+']'
        for localName in self.localNames:
            s = s+'/'+localName
        return s


    class RootType:
        _dom_root = 1
        _id = 2
        _class = 3